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
    """Market data source using ccxt for OKX exchange."""

    def __init__(self, exchange_id: Optional[str] = None) -> None:
        self._exchange_id = exchange_id or "okx"

    def _get_proxy_config(self) -> Dict[str, Any]:
        """获取代理配置"""
        proxy_url = (
            os.getenv("HTTPS_PROXY")
            or os.getenv("HTTP_PROXY")
            or os.getenv("https_proxy")
            or os.getenv("http_proxy")
            or "http://127.0.0.1:7890"
        )
        
        logger.info(f"🔧 [{self._exchange_id}] Using proxy: {proxy_url}")
        
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
                "defaultType": market_type,  # 关键：指定市场类型
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
        # 统一格式
        base_symbol = symbol.replace("-", "/")
        
        if market_type == "spot":
            return base_symbol  # BTC/USDT
        else:
            # 永续合约格式
            if ":" not in base_symbol:
                parts = base_symbol.split("/")
                if len(parts) == 2:
                    return f"{parts[0]}/{parts[1]}:{parts[1]}"  # BTC/USDT:USDT
            return base_symbol

    async def get_recent_candles(
        self, symbols: List[str], interval: str, lookback: int
    ) -> List[Candle]:
        """获取 K 线数据"""
        
        # OKX 不支持 1s
        if interval == "1s":
            logger.warning("1s fallback to 1m (OKX limit)")
            interval = "1m"

        all_candles: List[Candle] = []
        
        # 使用永续合约市场
        exchange = self._create_exchange(market_type="swap")
        
        try:
            # 只加载 swap 市场，避免加载 OPTION 等其他市场
            logger.debug(f"📡 Loading swap markets...")
            await exchange.load_markets()
            logger.debug(f"📡 Markets loaded: {len(exchange.markets)} pairs")
            
            for symbol in symbols:
                ccxt_symbol = self._get_ccxt_symbol(symbol, "swap")
                
                try:
                    # 检查交易对是否存在
                    if ccxt_symbol not in exchange.markets:
                        logger.warning(f"⚠️ Symbol {ccxt_symbol} not found in markets")
                        continue
                    
                    # 获取 OHLCV
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

    async def get_market_snapshot(self, symbols: List[str]) -> MarketSnapShotType:
        """获取市场快照（当前价格）"""
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