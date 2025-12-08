import asyncio
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from loguru import logger

from valuecell.agents.common.trading.models import (
    Candle,
    InstrumentRef,
    MarketSnapShotType,
)
from valuecell.agents.common.trading.utils import get_exchange_cls, normalize_symbol

from .interfaces import BaseMarketDataSource


class SimpleMarketDataSource(BaseMarketDataSource):
    """Market data source using ccxt for exchanges with multi-timeframe support."""

    # 默认多周期配置
    DEFAULT_TIMEFRAMES = {
        "1m": 120,   # 120 根 = 2 小时，用于入场时机
        "15m": 96,   # 96 根 = 24 小时，用于短期趋势
        "1h": 168,   # 168 根 = 7 天，用于中期趋势
        "4h": 180,   # 180 根 = 30 天，用于主趋势
        "1d": 90,    # 90 根 = 90 天，用于长期方向
    }

    def __init__(self, exchange_id: Optional[str] = None) -> None:
        self._exchange_id = exchange_id or "okx"
        self._markets_cache: Optional[Dict] = None
        self._cache_exchange: Optional[Any] = None

    def _get_proxy_config(self) -> Dict[str, Any]:
        """获取代理配置"""
        proxy_url = (
            os.getenv("HTTPS_PROXY")
            or os.getenv("HTTP_PROXY")
            or os.getenv("https_proxy")
            or os.getenv("http_proxy")
            or "http://127.0.0.1:7890"
        )
        
        return {
            "aiohttp_proxy": proxy_url,
            "proxies": {"http": proxy_url, "https": proxy_url},
            "timeout": 60000,
        }

    def _create_exchange(self, market_type: str = "swap") -> Any:
        """创建 exchange 实例
        
        Args:
            market_type: 'spot', 'swap', 'future'
        """
        exchange_cls = get_exchange_cls(self._exchange_id)
        
        config = {
            "enableRateLimit": True,
            "options": {
                "defaultType": market_type,
            },
            **self._get_proxy_config(),
        }
        
        return exchange_cls(config)

    def _get_ccxt_symbol(self, symbol: str, market_type: str = "swap") -> str:
        """转换为 ccxt 格式的交易对
        
        Args:
            symbol: 输入格式 'BTC/USDT' 或 'BTC-USDT'
            market_type: 'spot' 或 'swap'
            
        Returns:
            ccxt 格式: 'BTC/USDT' (spot) 或 'BTC/USDT:USDT' (swap)
        """
        base_symbol = symbol.replace("-", "/")
        
        if market_type == "spot":
            return base_symbol
        else:
            if ":" not in base_symbol:
                parts = base_symbol.split("/")
                if len(parts) == 2:
                    return f"{parts[0]}/{parts[1]}:{parts[1]}"
            return base_symbol

    async def get_recent_candles(
        self, symbols: List[str], interval: str, lookback: int
    ) -> List[Candle]:
        """获取单一周期 K 线数据（保持向后兼容）"""
        
        if interval == "1s":
            logger.warning("1s fallback to 1m (OKX limit)")
            interval = "1m"

        all_candles: List[Candle] = []
        exchange = self._create_exchange(market_type="swap")
        
        try:
            logger.debug(f"📡 Loading markets for {interval}...")
            await exchange.load_markets()
            logger.debug(f"📡 Markets loaded: {len(exchange.markets)} pairs")
            
            for symbol in symbols:
                ccxt_symbol = self._get_ccxt_symbol(symbol, "swap")
                
                try:
                    if ccxt_symbol not in exchange.markets:
                        logger.warning(f"⚠️ Symbol {ccxt_symbol} not found")
                        continue
                    
                    raw = await exchange.fetch_ohlcv(
                        ccxt_symbol,
                        timeframe=interval,
                        limit=lookback,
                    )
                    
                    for row in raw:
                        ts, o, h, l, c, v = row
                        all_candles.append(
                            Candle(
                                ts=int(ts),
                                instrument=InstrumentRef(
                                    symbol=symbol,
                                    exchange_id=self._exchange_id,
                                ),
                                open=float(o),
                                high=float(h),
                                low=float(l),
                                close=float(c),
                                volume=float(v),
                                interval=interval,
                            )
                        )
                    
                    logger.debug(f"✅ [{interval}] {symbol}: {len(raw)} candles")
                    
                except Exception as e:
                    logger.warning(f"❌ [{interval}] {symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to load markets: {e}")
        finally:
            try:
                await exchange.close()
            except:
                pass

        logger.info(f"📊 Candles total: {len(all_candles)} for {symbols} [{interval}]")
        return all_candles

    async def get_multi_timeframe_candles(
        self,
        symbols: List[str],
        timeframes: Optional[Dict[str, int]] = None,
    ) -> Dict[str, List[Candle]]:
        """获取多周期 K 线数据
        
        Args:
            symbols: 交易对列表 ['BTC/USDT', 'ETH/USDT']
            timeframes: {周期: 数量} 字典，如 {"1m": 60, "1h": 168, "4h": 180, "1d": 90}
                       如果为 None，使用默认配置
            
        Returns:
            {周期: [Candle]} 字典
            
        Example:
            >>> result = await source.get_multi_timeframe_candles(
            ...     symbols=["BTC/USDT", "ETH/USDT"],
            ...     timeframes={"1h": 168, "4h": 180, "1d": 90}
            ... )
            >>> print(result.keys())  # dict_keys(['1h', '4h', '1d'])
        """
        if timeframes is None:
            timeframes = self.DEFAULT_TIMEFRAMES
            
        result: Dict[str, List[Candle]] = {}
        exchange = self._create_exchange(market_type="swap")
        
        try:
            logger.info(f"📡 Loading markets for multi-timeframe analysis...")
            await exchange.load_markets()
            logger.info(f"📡 Markets loaded: {len(exchange.markets)} pairs")
            
            for timeframe, limit in timeframes.items():
                candles: List[Candle] = []
                
                for symbol in symbols:
                    ccxt_symbol = self._get_ccxt_symbol(symbol, "swap")
                    
                    try:
                        if ccxt_symbol not in exchange.markets:
                            logger.warning(f"⚠️ [{timeframe}] {ccxt_symbol} not found")
                            continue
                        
                        raw = await exchange.fetch_ohlcv(
                            ccxt_symbol,
                            timeframe=timeframe,
                            limit=limit,
                        )
                        
                        for row in raw:
                            ts, o, h, l, c, v = row
                            candles.append(
                                Candle(
                                    ts=int(ts),
                                    instrument=InstrumentRef(
                                        symbol=symbol,
                                        exchange_id=self._exchange_id,
                                    ),
                                    open=float(o),
                                    high=float(h),
                                    low=float(l),
                                    close=float(c),
                                    volume=float(v),
                                    interval=timeframe,
                                )
                            )
                        
                        logger.debug(f"✅ [{timeframe}] {symbol}: {len(raw)} candles")
                        
                        # 添加小延迟避免限流
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        logger.warning(f"❌ [{timeframe}] {symbol}: {e}")
                
                result[timeframe] = candles
                logger.info(f"📊 [{timeframe}] Total: {len(candles)} candles for {len(symbols)} symbols")
                
        except Exception as e:
            logger.error(f"❌ Multi-timeframe fetch failed: {e}")
        finally:
            try:
                await exchange.close()
            except:
                pass
        
        # 汇总日志
        total_candles = sum(len(c) for c in result.values())
        logger.info(f"📊 Multi-TF complete: {total_candles} candles across {list(timeframes.keys())}")
        
        return result

    async def get_market_snapshot(self, symbols: List[str]) -> MarketSnapShotType:
        """获取市场快照（当前价格、资金费率等）"""
        snapshot: Dict[str, Dict[str, Any]] = defaultdict(dict)
        success_count = 0
        
        exchange = self._create_exchange(market_type="swap")
        
        try:
            await exchange.load_markets()
            
            for symbol in symbols:
                ccxt_symbol = self._get_ccxt_symbol(symbol, "swap")
                
                try:
                    ticker = await exchange.fetch_ticker(ccxt_symbol)
                    snapshot[symbol]["price"] = ticker
                    success_count += 1
                    logger.debug(f"✅ Ticker {symbol}: last={ticker.get('last')}")
                    
                    # 尝试获取资金费率
                    try:
                        fr = await exchange.fetch_funding_rate(ccxt_symbol)
                        snapshot[symbol]["funding_rate"] = fr
                    except:
                        pass
                    
                    # 尝试获取持仓量
                    try:
                        oi = await exchange.fetch_open_interest(ccxt_symbol)
                        snapshot[symbol]["open_interest"] = oi
                    except:
                        pass
                        
                except Exception as e:
                    logger.warning(f"❌ Ticker {symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Markets load failed: {e}")
        finally:
            try:
                await exchange.close()
            except:
                pass

        logger.info(f"📈 Snapshot OK: {success_count}/{len(symbols)}")
        return dict(snapshot)