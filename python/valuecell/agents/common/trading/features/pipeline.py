"""Feature pipeline abstractions for the strategy agent.

This module encapsulates the data-fetch and feature-computation steps used by
strategy runtimes. Introducing a dedicated pipeline object means the decision
coordinator no longer needs direct access to the market data source or feature
computer—everything is orchestrated by the pipeline.

Updated: Added multi-timeframe support for better trend analysis.
"""

from __future__ import annotations

import asyncio
import itertools
from typing import List, Optional, Dict, Any

from loguru import logger

from valuecell.agents.common.trading.models import (
    CandleConfig,
    FeaturesPipelineResult,
    FeatureVector,
    UserRequest,
)

from ..data.interfaces import BaseMarketDataSource
from ..data.market import SimpleMarketDataSource
from .candle import SimpleCandleFeatureComputer
from .interfaces import (
    BaseFeaturesPipeline,
    CandleBasedFeatureComputer,
)
from .market_snapshot import MarketSnapshotFeatureComputer


class DefaultFeaturesPipeline(BaseFeaturesPipeline):
    """Default pipeline using the simple data source and feature computer.
    
    Now supports multi-timeframe analysis for better trend detection.
    """

    # 多周期配置：用于趋势分析
    MULTI_TIMEFRAME_CONFIGS = [
        CandleConfig(interval="1m", lookback=120),   # 2小时数据，入场时机
        CandleConfig(interval="15m", lookback=96),   # 24小时数据，短期趋势
        CandleConfig(interval="1h", lookback=168),   # 7天数据，中期趋势
        CandleConfig(interval="4h", lookback=180),   # 30天数据，主趋势
        CandleConfig(interval="1d", lookback=90),    # 90天数据，长期方向
    ]

    def __init__(
        self,
        *,
        request: UserRequest,
        market_data_source: BaseMarketDataSource,
        candle_feature_computer: CandleBasedFeatureComputer,
        market_snapshot_computer: MarketSnapshotFeatureComputer,
        candle_configurations: Optional[List[CandleConfig]] = None,
        use_multi_timeframe: bool = True,  # 新增：是否使用多周期
    ) -> None:
        self._request = request
        self._market_data_source = market_data_source
        self._candle_feature_computer = candle_feature_computer
        self._symbols = list(dict.fromkeys(request.trading_config.symbols))
        self._market_snapshot_computer = market_snapshot_computer
        self._use_multi_timeframe = use_multi_timeframe
        
        # 根据配置选择使用多周期还是原来的配置
        if use_multi_timeframe:
            self._candle_configurations = self.MULTI_TIMEFRAME_CONFIGS
            logger.info(f"📊 Using multi-timeframe analysis: {[c.interval for c in self._candle_configurations]}")
        else:
            self._candle_configurations = candle_configurations or [
                CandleConfig(interval="1s", lookback=60 * 3),
                CandleConfig(interval="1m", lookback=60 * 4),
            ]

    async def build(self) -> FeaturesPipelineResult:
        """
        Fetch candles and market snapshot, compute feature vectors concurrently,
        and combine results.
        
        With multi-timeframe enabled, fetches data across multiple timeframes
        for comprehensive trend analysis.
        """

        async def _fetch_candles(interval: str, lookback: int) -> List[FeatureVector]:
            """Fetches candles and computes features for a single (interval, lookback) pair."""
            try:
                _candles = await self._market_data_source.get_recent_candles(
                    self._symbols, interval, lookback
                )
                if not _candles:
                    logger.warning(f"⚠️ No candles returned for [{interval}]")
                    return []
                return self._candle_feature_computer.compute_features(candles=_candles)
            except Exception as e:
                logger.error(f"❌ Failed to fetch [{interval}] candles: {e}")
                return []

        async def _fetch_market_features() -> List[FeatureVector]:
            """Fetches market snapshot for all symbols and computes features."""
            try:
                market_snapshot = await self._market_data_source.get_market_snapshot(
                    self._symbols
                )
                market_snapshot = market_snapshot or {}
                return self._market_snapshot_computer.build(
                    market_snapshot, self._request.exchange_config.exchange_id
                )
            except Exception as e:
                logger.error(f"❌ Failed to fetch market snapshot: {e}")
                return []

        timeframes_str = [c.interval for c in self._candle_configurations]
        logger.info(
            f"📊 Starting concurrent data fetching for {len(self._candle_configurations)} "
            f"timeframes {timeframes_str} and market snapshot..."
        )
        
        # 创建所有任务
        tasks = [
            _fetch_candles(config.interval, config.lookback)
            for config in self._candle_configurations
        ]
        tasks.append(_fetch_market_features())

        # 并发执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果，过滤异常
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Task {i} failed with exception: {result}")
                valid_results.append([])
            else:
                valid_results.append(result)
        
        logger.info("✅ Concurrent data fetching complete.")

        # 最后一个是 market_features
        market_features: List[FeatureVector] = valid_results.pop()

        # 展平所有 candle features
        candle_features: List[FeatureVector] = list(
            itertools.chain.from_iterable(valid_results)
        )

        # 添加多周期综合分析（如果启用）
        if self._use_multi_timeframe and candle_features:
            multi_tf_summary = self._compute_multi_timeframe_summary(candle_features)
            logger.info(f"📈 Multi-TF Summary: {multi_tf_summary}")

        candle_features.extend(market_features)

        # 日志统计
        total_features = len(candle_features)
        by_interval = {}
        for f in candle_features:
            interval = getattr(f, 'interval', 'snapshot')
            by_interval[interval] = by_interval.get(interval, 0) + 1
        logger.info(f"📊 Total features: {total_features}, by interval: {by_interval}")

        return FeaturesPipelineResult(features=candle_features)

    def _compute_multi_timeframe_summary(
        self, 
        features: List[FeatureVector]
    ) -> Dict[str, Any]:
        """计算多周期趋势摘要
        
        分析各周期的趋势方向，输出综合信号。
        """
        summary = {
            "timeframes_analyzed": [],
            "trend_alignment": "unknown",
            "signals": {},
        }
        
        # 按周期分组
        by_interval: Dict[str, List[FeatureVector]] = {}
        for f in features:
            interval = getattr(f, 'interval', None)
            if interval:
                if interval not in by_interval:
                    by_interval[interval] = []
                by_interval[interval].append(f)
        
        summary["timeframes_analyzed"] = list(by_interval.keys())
        
        # 分析趋势一致性
        # 这里可以扩展更复杂的逻辑
        bullish_count = 0
        bearish_count = 0
        
        for interval, interval_features in by_interval.items():
            # 简化：检查该周期内的趋势特征
            for f in interval_features:
                # 假设 FeatureVector 有 trend 属性
                trend = getattr(f, 'trend', None)
                if trend == 'bullish':
                    bullish_count += 1
                elif trend == 'bearish':
                    bearish_count += 1
        
        if bullish_count > bearish_count * 2:
            summary["trend_alignment"] = "strong_bullish"
        elif bearish_count > bullish_count * 2:
            summary["trend_alignment"] = "strong_bearish"
        elif bullish_count > bearish_count:
            summary["trend_alignment"] = "bullish"
        elif bearish_count > bullish_count:
            summary["trend_alignment"] = "bearish"
        else:
            summary["trend_alignment"] = "neutral"
        
        return summary

    @classmethod
    def from_request(
        cls, 
        request: UserRequest,
        use_multi_timeframe: bool = True,  # 新增参数
    ) -> DefaultFeaturesPipeline:
        """Factory creating the default pipeline from a user request.
        
        Args:
            request: User request configuration
            use_multi_timeframe: If True, use multi-timeframe analysis (1m, 15m, 1h, 4h, 1d)
                                If False, use original 1s/1m configuration
        """
        market_data_source = SimpleMarketDataSource(
            exchange_id=request.exchange_config.exchange_id
        )
        candle_feature_computer = SimpleCandleFeatureComputer()
        market_snapshot_computer = MarketSnapshotFeatureComputer()
        return cls(
            request=request,
            market_data_source=market_data_source,
            candle_feature_computer=candle_feature_computer,
            market_snapshot_computer=market_snapshot_computer,
            use_multi_timeframe=use_multi_timeframe,
        )