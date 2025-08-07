import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import os
import dill
from scipy import signal
from scipy.spatial.distance import pdist
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

class AlloraMLWorkflow:
    def __init__(self, data_api_key, tickers, hours_needed, number_of_input_candles, target_length):
        self.api_key = data_api_key
        self.tickers = tickers
        self.hours_needed = hours_needed  # For input window
        self.number_of_input_candles = number_of_input_candles
        self.target_length = target_length  # Target horizon in hours
        self.test_targets = None

    def compute_from_date(self, extra_hours: int = 12) -> str:
        total_hours = self.hours_needed + extra_hours
        cutoff_time = datetime.utcnow() - timedelta(hours=total_hours)
        return cutoff_time.strftime("%Y-%m-%d")

    def list_ready_buckets(self, ticker, from_month):
        url = "https://api.upshot.xyz/v2/allora/market-data/ohlc/buckets/by-month"
        headers = {"x-api-key": self.api_key}
        resp = requests.get(url, headers=headers, params={"tickers": ticker, "from_month": from_month}, timeout=30)
        resp.raise_for_status()
        buckets = resp.json()["data"]["data"]
        return [b for b in buckets if b["state"] == "ready"]

    def fetch_bucket_csv(self, download_url):
        return pd.read_csv(download_url)

    def fetch_ohlcv_data(self, ticker, from_date: str, max_pages: int = 1000, sleep_sec: float = 0.1) -> pd.DataFrame:
        url = "https://api.upshot.xyz/v2/allora/market-data/ohlc"
        headers = {"x-api-key": self.api_key}
        params = {"tickers": ticker, "from_date": from_date}

        all_data = []
        pages_fetched = 0

        while pages_fetched < max_pages:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            payload = response.json()
            if not payload.get("status", False):
                raise RuntimeError("API responded with an error status.")

            all_data.extend(payload["data"]["data"])

            token = payload["data"].get("continuation_token")
            if not token:
                break

            params["continuation_token"] = token
            pages_fetched += 1
            time.sleep(sleep_sec)

        df = pd.DataFrame(all_data)
        if df.empty:
            raise ValueError("No data returned from API.")

        for col in ["open", "high", "low", "close", "volume", "volume_notional"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        return df

    def create_5_min_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index("date").sort_index()
        return df.resample("5min").apply({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "trades_done": "sum"
        })

    def compute_target(self, df: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
        df["future_close"] = df["close"].shift(freq=f"-{hours}h")
        df["target"] = np.log(df["future_close"]) - np.log(df["close"])
        return df

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        bb_width = (upper_band - lower_band) / sma
        return upper_band, lower_band, bb_position, bb_width

    def calculate_macd(self, prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast_period).mean()
        ema_slow = prices.ewm(span=slow_period).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent

    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute comprehensive technical indicators for the OHLCV data
        """
        df = df.copy()
        
        # Price returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close']).diff()
        
        # Moving Averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
            df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}']
        
        # Price momentum and velocity
        for period in [1, 3, 5, 10]:
            df[f'price_change_{period}'] = df['close'].pct_change(periods=period)
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # RSI
        df['rsi_14'] = self.calculate_rsi(df['close'], 14)
        df['rsi_7'] = self.calculate_rsi(df['close'], 7)
        
        # Bollinger Bands
        bb_upper, bb_lower, bb_position, bb_width = self.calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        df['bb_position'] = bb_position  # %B indicator
        df['bb_width'] = bb_width
        
        # MACD
        macd_line, signal_line, histogram = self.calculate_macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # Stochastic
        stoch_k, stoch_d = self.calculate_stochastic(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        # Volatility measures
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
            df[f'log_volatility_{period}'] = df['log_returns'].rolling(period).std()
        
        # Average True Range (ATR)
        df['tr'] = np.maximum(df['high'] - df['low'], 
                             np.maximum(abs(df['high'] - df['close'].shift(1)),
                                       abs(df['low'] - df['close'].shift(1))))
        df['atr_14'] = df['tr'].rolling(14).mean()
        df['atr_7'] = df['tr'].rolling(7).mean()
        
        # Price range features
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['co_ratio'] = (df['close'] - df['open']) / df['open']
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Volume indicators
        for period in [5, 10, 20]:
            df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['returns']) * df['volume']).cumsum()
        df['obv_sma_10'] = df['obv'].rolling(10).mean()
        df['obv_ratio'] = df['obv'] / df['obv_sma_10']
        
        # Volume-Price Trend (VPT)
        df['vpt'] = (df['volume'] * df['returns']).cumsum()
        df['vpt_sma_10'] = df['vpt'].rolling(10).mean()
        
        # Price-Volume correlation
        for period in [10, 20]:
            df[f'price_volume_corr_{period}'] = df['close'].rolling(period).corr(df['volume'])
        
        # Higher highs, lower lows
        for period in [5, 10, 20]:
            df[f'higher_high_{period}'] = (df['high'] > df['high'].rolling(period).max().shift(1)).astype(int)
            df[f'lower_low_{period}'] = (df['low'] < df['low'].rolling(period).min().shift(1)).astype(int)
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Market session indicators (assuming 24/7 crypto, but can be useful)
        df['market_open_distance'] = np.minimum(df['hour'], 24 - df['hour'])
        
        return df

    def calculate_hurst_exponent(self, prices: pd.Series, max_lag: int = 100) -> float:
        """
        Calculate Hurst Exponent using R/S analysis
        H > 0.5: trending (persistent)
        H < 0.5: mean-reverting (anti-persistent)
        H = 0.5: random walk
        """
        if len(prices) < max_lag * 2:
            return np.nan
            
        lags = range(2, min(max_lag, len(prices) // 4))
        rs_values = []
        
        log_prices = np.log(prices.dropna())
        
        for lag in lags:
            # Divide series into non-overlapping periods
            n_periods = len(log_prices) // lag
            if n_periods < 2:
                continue
                
            rs_period = []
            for i in range(n_periods):
                period_data = log_prices[i*lag:(i+1)*lag]
                if len(period_data) < lag:
                    continue
                    
                # Calculate mean
                mean_return = np.mean(np.diff(period_data))
                
                # Calculate cumulative deviations
                deviations = np.diff(period_data) - mean_return
                cumulative_deviations = np.cumsum(deviations)
                
                # Calculate range
                R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
                
                # Calculate standard deviation
                S = np.std(np.diff(period_data))
                
                if S > 0:
                    rs_period.append(R / S)
            
            if rs_period:
                rs_values.append(np.mean(rs_period))
        
        if len(rs_values) < 3:
            return np.nan
            
        # Linear regression of log(R/S) vs log(lag)
        log_lags = np.log(lags[:len(rs_values)])
        log_rs = np.log(rs_values)
        
        # Remove infinite or NaN values
        mask = np.isfinite(log_rs) & np.isfinite(log_lags)
        if np.sum(mask) < 3:
            return np.nan
            
        hurst = np.polyfit(log_lags[mask], log_rs[mask], 1)[0]
        return max(0, min(1, hurst))  # Clamp between 0 and 1

    def calculate_permutation_entropy(self, series: pd.Series, order: int = 3, normalize: bool = True) -> float:
        """
        Calculate Permutation Entropy - measures complexity of ordinal patterns
        Higher values indicate more complex/random patterns
        """
        if len(series) < order + 1:
            return np.nan
            
        series = series.dropna().values
        if len(series) < order + 1:
            return np.nan
        
        # Create ordinal patterns
        ordinal_patterns = []
        for i in range(len(series) - order + 1):
            window = series[i:i + order]
            # Get permutation pattern (ranking)
            pattern = tuple(np.argsort(np.argsort(window)))
            ordinal_patterns.append(pattern)
        
        # Count pattern frequencies
        from collections import Counter
        pattern_counts = Counter(ordinal_patterns)
        
        # Calculate relative frequencies
        total_patterns = len(ordinal_patterns)
        probabilities = [count / total_patterns for count in pattern_counts.values()]
        
        # Calculate entropy
        pe = entropy(probabilities, base=2)
        
        if normalize:
            # Normalize by maximum possible entropy
            max_entropy = np.log2(np.math.factorial(order))
            pe = pe / max_entropy if max_entropy > 0 else 0
            
        return pe

    def calculate_fractal_dimension(self, prices: pd.Series, method: str = 'higuchi') -> float:
        """
        Calculate Fractal Dimension using Higuchi's method
        FD close to 1: trend-like behavior
        FD close to 2: random walk-like behavior
        """
        if len(prices) < 50:
            return np.nan
            
        prices = prices.dropna().values
        if len(prices) < 50:
            return np.nan
            
        N = len(prices)
        k_max = min(20, N // 4)
        
        Lk = []
        k_values = []
        
        for k in range(1, k_max + 1):
            Lm = []
            for m in range(k):
                # Construct subsequence
                max_i = int((N - m - 1) / k)
                if max_i < 1:
                    continue
                    
                length = 0
                for i in range(1, max_i + 1):
                    length += abs(prices[m + i * k] - prices[m + (i - 1) * k])
                
                # Normalize
                length = length * (N - 1) / (max_i * k)
                Lm.append(length)
            
            if Lm:
                Lk.append(np.mean(Lm))
                k_values.append(k)
        
        if len(Lk) < 3:
            return np.nan
            
        # Linear regression in log-log space
        log_k = np.log(k_values)
        log_Lk = np.log(Lk)
        
        # Remove infinite values
        mask = np.isfinite(log_Lk) & np.isfinite(log_k)
        if np.sum(mask) < 3:
            return np.nan
            
        slope = np.polyfit(log_k[mask], log_Lk[mask], 1)[0]
        fractal_dim = -slope
        
        return max(1, min(2, fractal_dim))  # Clamp between 1 and 2

    def calculate_sample_entropy(self, series: pd.Series, m: int = 2, r: float = None) -> float:
        """
        Calculate Sample Entropy - measures time series regularity
        Lower values indicate more regular/predictable patterns
        """
        if len(series) < 10:
            return np.nan
            
        series = series.dropna().values
        if len(series) < 10:
            return np.nan
            
        N = len(series)
        
        if r is None:
            r = 0.2 * np.std(series)
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = []
            for i in range(N - m + 1):
                patterns.append(series[i:i + m])
            
            phi = 0
            for i in range(len(patterns)):
                matches = 0
                for j in range(len(patterns)):
                    if i != j and _maxdist(patterns[i], patterns[j], m) <= r:
                        matches += 1
                if matches > 0:
                    phi += np.log(matches / (N - m))
            
            return phi / (N - m + 1)
        
        try:
            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)
            return phi_m - phi_m1
        except:
            return np.nan

    def calculate_wavelet_features(self, prices: pd.Series, wavelet: str = 'db4', levels: int = 4) -> dict:
        """
        Calculate Wavelet decomposition features
        Returns energy distribution across frequency bands
        """
        try:
            import pywt
        except ImportError:
            # Fallback: simple frequency domain analysis
            return self._calculate_fft_features(prices, levels)
        
        if len(prices) < 2**levels:
            return {f'wavelet_energy_level_{i}': np.nan for i in range(levels)}
            
        prices_clean = prices.dropna().values
        if len(prices_clean) < 2**levels:
            return {f'wavelet_energy_level_{i}': np.nan for i in range(levels)}
        
        try:
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(prices_clean, wavelet, level=levels)
            
            # Calculate energy at each level
            energies = {}
            total_energy = sum([np.sum(c**2) for c in coeffs])
            
            for i, c in enumerate(coeffs):
                energy = np.sum(c**2)
                energies[f'wavelet_energy_level_{i}'] = energy / total_energy if total_energy > 0 else 0
                
            return energies
        except:
            return {f'wavelet_energy_level_{i}': np.nan for i in range(levels)}

    def _calculate_fft_features(self, prices: pd.Series, levels: int = 4) -> dict:
        """
        Fallback frequency domain analysis using FFT
        """
        if len(prices) < 10:
            return {f'wavelet_energy_level_{i}': np.nan for i in range(levels)}
            
        prices_clean = prices.dropna().values
        if len(prices_clean) < 10:
            return {f'wavelet_energy_level_{i}': np.nan for i in range(levels)}
            
        # Calculate FFT
        fft_vals = np.fft.fft(prices_clean)
        power_spectrum = np.abs(fft_vals)**2
        
        # Divide into frequency bands
        n = len(power_spectrum) // 2  # Use only positive frequencies
        band_size = n // levels
        
        energies = {}
        total_energy = np.sum(power_spectrum[:n])
        
        for i in range(levels):
            start_idx = i * band_size
            end_idx = (i + 1) * band_size if i < levels - 1 else n
            band_energy = np.sum(power_spectrum[start_idx:end_idx])
            energies[f'wavelet_energy_level_{i}'] = band_energy / total_energy if total_energy > 0 else 0
            
        return energies

    def calculate_recurrence_features(self, prices: pd.Series, embedding_dim: int = 3, delay: int = 1, radius: float = None) -> dict:
        """
        Calculate Recurrence Quantification Analysis features
        Measures deterministic structures in time series
        """
        if len(prices) < embedding_dim * delay + 10:
            return {
                'recurrence_rate': np.nan,
                'determinism': np.nan,
                'laminarity': np.nan,
                'entropy_rqa': np.nan
            }
            
        prices_clean = prices.dropna().values
        if len(prices_clean) < embedding_dim * delay + 10:
            return {
                'recurrence_rate': np.nan,
                'determinism': np.nan,
                'laminarity': np.nan,
                'entropy_rqa': np.nan
            }
        
        # Create time-delay embedding
        N = len(prices_clean)
        M = N - (embedding_dim - 1) * delay
        
        if M <= 0:
            return {
                'recurrence_rate': np.nan,
                'determinism': np.nan,
                'laminarity': np.nan,
                'entropy_rqa': np.nan
            }
        
        # Embedded vectors
        embedded = np.zeros((M, embedding_dim))
        for i in range(M):
            for j in range(embedding_dim):
                embedded[i, j] = prices_clean[i + j * delay]
        
        # Calculate distance matrix
        distances = pdist(embedded, metric='euclidean')
        
        if radius is None:
            radius = 0.1 * np.std(distances)
        
        # Create recurrence matrix (upper triangular)
        from scipy.spatial.distance import squareform
        dist_matrix = squareform(distances)
        recurrence_matrix = (dist_matrix <= radius).astype(int)
        
        # Calculate RQA measures
        total_points = M * (M - 1) // 2
        recurrent_points = np.sum(recurrence_matrix[np.triu_indices(M, k=1)])
        
        # Recurrence Rate
        rr = recurrent_points / total_points if total_points > 0 else 0
        
        # Determinism and Laminarity (simplified calculation)
        try:
            # Count diagonal lines (determinism)
            diag_lines = []
            for k in range(1, min(M, 20)):  # Check up to 20 diagonals
                diag = np.diag(recurrence_matrix, k=k)
                if len(diag) > 2:
                    # Find consecutive ones
                    consecutive = 0
                    max_consecutive = 0
                    for val in diag:
                        if val == 1:
                            consecutive += 1
                            max_consecutive = max(max_consecutive, consecutive)
                        else:
                            consecutive = 0
                    if max_consecutive >= 2:
                        diag_lines.append(max_consecutive)
            
            det = len(diag_lines) / max(1, M - 1)
            
            # Simplified laminarity
            lam = np.mean(diag_lines) / M if diag_lines else 0
            
            # Entropy (simplified)
            if diag_lines:
                hist, _ = np.histogram(diag_lines, bins=min(10, len(diag_lines)))
                prob = hist / np.sum(hist)
                ent = entropy(prob[prob > 0], base=2)
            else:
                ent = 0
                
        except:
            det, lam, ent = 0, 0, 0
        
        return {
            'recurrence_rate': rr,
            'determinism': det,
            'laminarity': lam,
            'entropy_rqa': ent
        }

    def compute_esoteric_features(self, df: pd.DataFrame, window_size: int = 100) -> pd.DataFrame:
        """
        Compute esoteric features using rolling windows
        """
        df = df.copy()
        
        print("Computing esoteric features...")
        
        # Initialize feature columns
        esoteric_features = [
            'hurst_exponent', 'permutation_entropy', 'fractal_dimension', 'sample_entropy',
            'recurrence_rate', 'determinism', 'laminarity', 'entropy_rqa'
        ]
        
        # Add wavelet features
        for i in range(4):
            esoteric_features.append(f'wavelet_energy_level_{i}')
        
        # Initialize with NaN
        for feature in esoteric_features:
            df[feature] = np.nan
        
        # Calculate features using rolling windows
        close_prices = df['close']
        
        for i in range(window_size, len(df)):
            window_data = close_prices.iloc[i-window_size:i]
            
            if len(window_data.dropna()) < window_size * 0.8:  # Require 80% valid data
                continue
            
            try:
                # Hurst Exponent
                df.iloc[i, df.columns.get_loc('hurst_exponent')] = self.calculate_hurst_exponent(window_data)
                
                # Permutation Entropy
                df.iloc[i, df.columns.get_loc('permutation_entropy')] = self.calculate_permutation_entropy(window_data)
                
                # Fractal Dimension
                df.iloc[i, df.columns.get_loc('fractal_dimension')] = self.calculate_fractal_dimension(window_data)
                
                # Sample Entropy
                df.iloc[i, df.columns.get_loc('sample_entropy')] = self.calculate_sample_entropy(window_data)
                
                # Wavelet Features
                wavelet_features = self.calculate_wavelet_features(window_data)
                for feature_name, value in wavelet_features.items():
                    if feature_name in df.columns:
                        df.iloc[i, df.columns.get_loc(feature_name)] = value
                
                # Recurrence Features
                recurrence_features = self.calculate_recurrence_features(window_data)
                for feature_name, value in recurrence_features.items():
                    if feature_name in df.columns:
                        df.iloc[i, df.columns.get_loc(feature_name)] = value
                        
            except Exception as e:
                # Skip problematic windows
                continue
        
        print("Esoteric features computation completed.")
        return df

    def extract_rolling_daily_features(
        self, data: pd.DataFrame, lookback: int, number_of_candles: int, start_times: list
    ) -> pd.DataFrame:
        # Convert index to array for quick lookup
        ts_index = data.index.to_numpy()
        data_values = data[["open", "high", "low", "close", "volume", "trades_done"]].to_numpy()
        features_list = []
        index_list = []
    
        candle_length = lookback * 12  # 12 points per hour if 5min bars
    
        for T in start_times:
            # Find the last index <= T
            pos = np.searchsorted(ts_index, T, side="right") - 1
            if pos - candle_length < 0:
                continue
    
            window = data_values[pos - candle_length:pos]
    
            # Group window into number_of_candles equal chunks
            try:
                reshaped = window.reshape(number_of_candles, -1, 6)
            except ValueError:
                continue  # Skip if window can't be reshaped
    
            open_ = reshaped[:, 0, 0]
            high_ = reshaped[:, :, 1].max(axis=1)
            low_ = reshaped[:, :, 2].min(axis=1)
            close_ = reshaped[:, -1, 3]
            volume_ = reshaped[:, :, 4].sum(axis=1)
            trades_ = reshaped[:, :, 5].sum(axis=1)
    
            last_close = close_[-1]
            last_volume = volume_[-1]
            if last_close == 0 or np.isnan(last_close) or last_volume == 0 or np.isnan(last_volume):
                continue
    
            features = np.stack([open_, high_, low_, close_, volume_, trades_], axis=1)
            features[:, :4] /= last_close  # Normalize OHLC
            features[:, 4] /= last_volume  # Normalize volume
    
            features_list.append(features.flatten())
            index_list.append(T)
    
        if not features_list:
            return pd.DataFrame(columns=[
                f"feature_{f}_{i}" for i in range(number_of_candles) for f in ["open", "high", "low", "close", "volume", "trades_done"]
            ])
    
        features_array = np.vstack(features_list)
        columns = [f"feature_{f}_{i}" for i in range(number_of_candles) for f in ["open", "high", "low", "close", "volume", "trades_done"]]
        return pd.DataFrame(features_array, index=index_list, columns=columns)

    def get_live_features(self, ticker):
        from_date = self.compute_from_date()
        df = self.fetch_ohlcv_data(ticker, from_date)
        five_min_bars = self.create_5_min_bars(df)
        five_min_bars = self.compute_technical_indicators(five_min_bars)
        five_min_bars = self.compute_esoteric_features(five_min_bars)
        if len(five_min_bars) < self.hours_needed * 12:
            raise ValueError("Not enough historical data.")
        live_time = five_min_bars.index[-2]
        features = self.extract_rolling_daily_features(five_min_bars, self.hours_needed, self.number_of_input_candles, [live_time])
        if features.empty:
            raise ValueError("No features returned.")
        return features

    def evaluate_test_data(self, predictions: pd.Series) -> dict:
        if self.test_targets is None:
            raise ValueError("Test targets not set. Run get_train_validation_test_data first.")

        if not predictions.index.equals(self.test_targets.index):
            raise ValueError("Prediction index must match test target index.")

        y_true = self.test_targets
        y_pred = predictions

        corr = np.corrcoef(y_true, y_pred)[0, 1]
        directional_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))

        return {
            "correlation": corr,
            "directional_accuracy": directional_accuracy
        }

    def get_full_feature_target_dataframe(self, from_month="2025-01") -> pd.DataFrame:
        """
        Returns a DataFrame containing all features and target values for all tickers,
        with a MultiIndex of (date, ticker). Does not split into training/validation.
        """
        all_data = {}
        for t in self.tickers:
            print(f"Downloading Historical Data for {t}")
            frames = []
            for bucket in self.list_ready_buckets(t, from_month):
                df = self.fetch_bucket_csv(bucket["download_url"])
                df["bucket_start"] = bucket["start"]
                df["bucket_end"] = bucket["end"]
                df["availability"] = bucket["availability"]
                frames.append(df)
            combined_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if not combined_df.empty:
                latest_ts = sorted(pd.to_datetime(combined_df["date"]).dt.date.unique())[-2]
                live_df = self.fetch_ohlcv_data(t, latest_ts.strftime("%Y-%m-%d"))
                combined_df = pd.concat([combined_df, live_df], ignore_index=True)
                combined_df["date"] = pd.to_datetime(combined_df["date"])
                combined_df = combined_df.drop_duplicates(subset="date")
            all_data[t] = combined_df
    
        datasets = []
        for t in self.tickers:
            print(f"Processing 5-minute bars for {t}")
            df = self.create_5_min_bars(all_data[t])
            print(f"Computing technical indicators")
            df = self.compute_technical_indicators(df)
            print(f"Computing esoteric features")
            df = self.compute_esoteric_features(df)
            df = self.compute_target(df, self.target_length)
            rolling_features = self.extract_rolling_daily_features(
                df, self.hours_needed, self.number_of_input_candles, df.index.tolist()
            )
            df = df.join(rolling_features, how='inner')
            df["ticker"] = t
            datasets.append(df)
    
        full_data = pd.concat(datasets).sort_index()
        print(f"get_full_feature_target_dataframe - Total columns: {len(full_data.columns)}")
        
        # Create MultiIndex from existing index (date) and ticker column
        reset_df = full_data.reset_index()
        date_col = None
        for col in reset_df.columns:
            if col in ['date', 'index'] or 'date' in col.lower():
                date_col = col
                break
        
        if date_col is None:
            raise ValueError(f"Could not find date column in: {list(reset_df.columns)}")
        
        full_data.index = pd.MultiIndex.from_frame(reset_df[[date_col, "ticker"]])
        
        # Only drop rows where target or essential features are missing
        essential_columns = ['target', 'open', 'high', 'low', 'close', 'volume']
        full_data = full_data.dropna(subset=essential_columns)
    
        return full_data

    def get_train_validation_test_data(self, from_month="2025-01", validation_months=3, test_months=3, force_redownload=False):
        def generate_filename():
            """Generate a unique filename based on parameters."""
            tickers_str = "_".join(self.tickers)
            return f"data_{tickers_str}_{from_month}_val{validation_months}_test{test_months}.pkl"

        def save_to_disk(data, filename):
            """Save data to disk."""
            with open(filename, "wb") as f:
                dill.dump(data, f)

        def load_from_disk(filename):
            """Load data from disk."""
            with open(filename, "rb") as f:
                X_train, y_train, X_val, y_val, X_test, y_test = dill.load(f)
            self.test_targets = y_test
            return X_train, y_train, X_val, y_val, X_test, y_test

        # Generate the filename
        filename = generate_filename()

        # Check if the file exists and load it if not forcing a redownload
        if os.path.exists(filename) and not force_redownload:
            print(f"Loading data from {filename}")
            return load_from_disk(filename)

        # If file doesn't exist or force_redownload is True, proceed with data preparation
        all_data = {}
        for t in self.tickers:
            print(f"Downloading Historical Data for {t}")
            frames = []
            for bucket in self.list_ready_buckets(t, from_month):
                df = self.fetch_bucket_csv(bucket["download_url"])
                df["bucket_start"] = bucket["start"]
                df["bucket_end"] = bucket["end"]
                df["availability"] = bucket["availability"]
                frames.append(df)
            combined_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if not combined_df.empty:
                latest_ts = sorted(pd.to_datetime(combined_df["date"]).dt.date.unique())[-2]
                live_df = self.fetch_ohlcv_data(t, latest_ts.strftime("%Y-%m-%d"))
                combined_df = pd.concat([combined_df, live_df], ignore_index=True)
                combined_df['date'] = pd.to_datetime(combined_df['date'])
                combined_df = combined_df.drop_duplicates(subset='date')
            all_data[t] = combined_df

        datasets = []
        for t in self.tickers:
            print(f"Processing 5-minute bars for {t}")
            df = self.create_5_min_bars(all_data[t])
            print(f"Computing technical indicators")
            df = self.compute_technical_indicators(df)
            print(f"After technical indicators: {len(df.columns)} columns")
            print(f"Sample tech columns: {[col for col in df.columns if col in ['returns', 'sma_5', 'rsi_14', 'macd']]}")
            
            print(f"Computing esoteric features")
            df = self.compute_esoteric_features(df)
            print(f"After esoteric features: {len(df.columns)} columns")
            print(f"Sample esoteric: {[col for col in df.columns if col in ['hurst_exponent', 'permutation_entropy']]}")
            
            print(f"Computing target")
            df = self.compute_target(df, self.target_length)
            print(f"Extracting rolling candle features")
            rolling_features = self.extract_rolling_daily_features(df, self.hours_needed, self.number_of_input_candles, df.index.tolist())
            
            # Combine technical indicators, esoteric features, and rolling features
            df = df.join(rolling_features, how='inner')
            df["ticker"] = t
            
            print(f"Total columns for {t}: {len(df.columns)}")
            print(f"Sample columns: {list(df.columns[:10])}...")
            datasets.append(df)

        full_data = pd.concat(datasets).sort_index()
        print(f"Before MultiIndex - Total columns: {len(full_data.columns)}")
        print(f"Sample columns: {list(full_data.columns[:15])}...")
        
        # Create MultiIndex from existing index (date) and ticker column
        reset_df = full_data.reset_index()
        print(f"Reset index columns: {list(reset_df.columns)}")
        
        # Handle different possible index names
        date_col = None
        for col in reset_df.columns:
            if col in ['date', 'index'] or 'date' in col.lower():
                date_col = col
                break
        
        if date_col is None:
            raise ValueError(f"Could not find date column in: {list(reset_df.columns)}")
        
        full_data.index = pd.MultiIndex.from_frame(reset_df[[date_col, "ticker"]])
        print(f"Before dropna - Shape: {full_data.shape}")
        print(f"Before dropna - Columns: {len(full_data.columns)}")
        
        # Only drop rows where target or essential features are missing
        # Keep technical indicators even if some esoteric features are NaN
        essential_columns = ['target', 'open', 'high', 'low', 'close', 'volume']
        full_data = full_data.dropna(subset=essential_columns)
        print(f"After selective dropna - Shape: {full_data.shape}")
        print(f"Final columns: {len(full_data.columns)}")
        
        # Show sample of non-basic features
        basic_features = ['open', 'high', 'low', 'close', 'volume', 'trades_done', 'ticker', 'target', 'future_close']
        feature_cols = [col for col in full_data.columns if not col.startswith('feature_') and col not in basic_features]
        print(f"Non-basic feature columns ({len(feature_cols)}): {feature_cols[:20]}...")

        # Define cutoff dates for test, validation, and training sets
        test_cutoff = datetime.utcnow() - pd.DateOffset(months=test_months)
        val_cutoff_start = test_cutoff - timedelta(hours=self.target_length) - pd.DateOffset(months=validation_months)
        val_cutoff_end = test_cutoff - timedelta(hours=self.target_length)
        train_cutoff = val_cutoff_start - timedelta(hours=self.target_length)

        # Create masks for each set
        # Use level 0 (first level) which is the date, regardless of its name
        date_level = full_data.index.get_level_values(0)
        test_mask = date_level >= str(test_cutoff)
        val_mask = (date_level >= str(val_cutoff_start)) & (date_level < str(val_cutoff_end))
        train_mask = date_level < str(train_cutoff)

        # Store validation targets for evaluation
        self.validation_targets = full_data.loc[val_mask, ["target"]]

        # Split data into train, validation, and test sets
        X_train = full_data.loc[train_mask].drop(columns=["target", "future_close"])
        y_train = full_data.loc[train_mask]["target"]
        X_val = full_data.loc[val_mask].drop(columns=["target", "future_close"])
        y_val = full_data.loc[val_mask]["target"]
        X_test = full_data.loc[test_mask].drop(columns=["target", "future_close"])
        y_test = full_data.loc[test_mask]["target"]

        self.test_targets = y_test

        # Save the prepared data to disk
        print(f"Saving data to {filename}")
        save_to_disk((X_train, y_train, X_val, y_val, X_test, y_test), filename)

        return X_train, y_train, X_val, y_val, X_test, y_test

