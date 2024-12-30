import rqdatac as rq
rq.init('pkulab','PKUlab2021','222.29.71.3:16010')
from backtesting import *
import pandas as pd
import os
import numpy as np


class DataLoader:
    """数据加载类,用于读取和处理双索引DataFrame数据"""
    
    def __init__(self, file_path='模拟cache存储数据.csv',index='000300.XSHG'):
        """
        初始化数据加载器
        参数:
            file_path: CSV文件路径
        """
        self.file_path = file_path
        self.data = None
        self.stock_pool = None
        self.index_id=index
        
    def load_data(self):
        """读取CSV文件并设置双索引"""
        try:
            # 读取CSV文件
            self.data = pd.read_csv(self.file_path)
            # 设置双索引
            self.data = self.data.set_index(['stk_id', 'date'])
            return True
        except Exception as e:
            print(f"数据加载错误: {e}")
            return False
            
    def generate_stock_pool(self):
        """
        生成股票池方法
        这里使用示例股票池,实际应用中可根据需求修改选股逻辑
        """
        # 示例:选取前100只股票
        if self.data is not None:
            unique_stocks = self.data.index.get_level_values('stk_id').unique()
            self.stock_pool = list(unique_stocks[:100])
            return self.stock_pool
        return []
        
    def generate_stock_pool_index(self):
        stock_id=rq.index_components(self.index_id, start_date='2020-01-02', 
                                     end_date='2020-01-02', market='cn',return_create_tm=False)[datetime(2020, 1, 2, 0, 0)]
        stock_id=rq.id_convert(stock_id,to='normal')
        self.stock_pool=stock_id
        return self.stock_pool
        
    def filter_data(self):
        """根据股票池筛选数据"""
        if self.data is None or self.stock_pool is None:
            print("请先加载数据并生成股票池")
            return None
            
        # 获取股票池和现有股票的交集
        valid_stocks = list(set(self.stock_pool) & 
                          set(self.data.index.get_level_values('stk_id').unique()))
                          
        # 筛选数据
        filtered_data = self.data.loc[valid_stocks]
        filtered_data.index = filtered_data.index.set_levels(pd.to_datetime(filtered_data.index.levels[1]), level=1)
        return filtered_data
        
    def process_data(self):
        """完整的数据处理流程"""
        if self.load_data():
            self.generate_stock_pool_index()
            return self.filter_data()
        return None



class StockData(PandasData):
    def __init__(self, stock_code, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stock_code = stock_code

# 双均线策略
class MAStrategy(Strategy):
    """双均线策略"""
    def __init__(self, broker, data, fast_period=5, slow_period=15):
        super().__init__(broker, data)
        self.fast_period = fast_period  # 短期均线周期
        self.slow_period = slow_period  # 长期均线周期
        self.fast_ma = {}  # 每只股票的短期均线
        self.slow_ma = {}  # 每只股票的长期均线
        self.positions = {}  # 每只股票的持仓
        self.broker = broker
        self.buy_signals = []  # 存储产生买入信号的股票信息
        super().__setattr__('broker', broker)
        
        
    def next(self):
        """策略逻辑"""
        self.buy_signals = []  # 清空买入信号列表
        
        # 遍历每只股票
        for stock_code, data in self.data.items():
            # 获取收盘价序列
            close_prices = list(data.lines['close'])
            
            # 初始化该股票的均线列表
            if stock_code not in self.fast_ma:
                self.fast_ma[stock_code] = []
                self.slow_ma[stock_code] = []
            
            # 计算均线
            if len(close_prices) >= self.slow_period:
                fast_ma = sum(close_prices[-self.fast_period:]) / self.fast_period
                slow_ma = sum(close_prices[-self.slow_period:]) / self.slow_period
                
                self.fast_ma[stock_code].append(fast_ma)
                self.slow_ma[stock_code].append(slow_ma)
                
                # 交易逻辑
                if len(self.fast_ma[stock_code]) >= 2:  # 确保有足够的均线数据
                    # 金叉买入信号
                    if (self.fast_ma[stock_code][-2] <= self.slow_ma[stock_code][-2] and 
                        self.fast_ma[stock_code][-1] > self.slow_ma[stock_code][-1] ):
                        self.buy_signals.append({
                            'stock_code': stock_code,
                            'price': close_prices[-1]
                        })
                        
                    # 死叉卖出    
                    elif (self.fast_ma[stock_code][-2] >= self.slow_ma[stock_code][-2] and 
                          self.fast_ma[stock_code][-1] < self.slow_ma[stock_code][-1] and 
                          stock_code in self.broker.positions):
                        size = self.broker.positions[stock_code]  # 获取持仓数量
                        #print(size)
                        if size > 0:
                            self.sell(size, close_prices[-1], stock_code)
        
        # 处理买入信号
        #print(self.broker.cash)
        if self.buy_signals:
            available_cash = self.broker.cash * 0.5  # 可用资金为总资金的80%
            per_stock_cash = available_cash / len(self.buy_signals)  # 每只股票分配的资金
            
            # 执行买入订单
            for signal in self.buy_signals:
                size = int(per_stock_cash / signal['price'])  # 计算可买入的股票数量
                if size > 0:
                    self.buy(size, signal['price'], signal['stock_code'])

class ReversalStrategy(Strategy):
    """N日反转策略"""
    def __init__(self, broker, data, n_days=5):
        super().__init__(broker, data)
        self.n_days = n_days  # N日周期
        self.returns = {}  # 每只股票的收益率序列
        self.buy_signals = []  # 存储产生买入信号的股票信息
        super().__setattr__('broker', broker)
        
    def next(self):
        """策略逻辑"""
        self.buy_signals = []  # 清空买入信号列表
        
        # 遍历每只股票
        for stock_code, data in self.data.items():
            # 获取收盘价序列
            close_prices = list(data.lines['close'])
            
            # 初始化该股票的收益率列表
            if stock_code not in self.returns:
                self.returns[stock_code] = []
            
            # 计算N日收益率
            if len(close_prices) >= self.n_days + 1:
                n_day_return = (close_prices[-1] - close_prices[-self.n_days-1]) / close_prices[-self.n_days-1]
                self.returns[stock_code].append(n_day_return)
                
                # 交易逻辑
                if len(self.returns[stock_code]) >= 2:
                    # 跌幅超过10%时买入
                    if n_day_return < -0.08 :
                        self.buy_signals.append({
                            'stock_code': stock_code,
                            'price': close_prices[-1]
                        })
                    
                    # 涨幅超过5%时卖出
                    elif n_day_return > 0.08 and stock_code in self.broker.positions:
                        size = self.broker.positions[stock_code]  # 获取持仓数量
                        if size > 0:
                            self.sell(size, close_prices[-1], stock_code)
        
        # 处理买入信号
        if self.buy_signals:
            available_cash = self.broker.cash * 0.5  # 可用资金为总资金的50%
            per_stock_cash = available_cash / len(self.buy_signals)  # 每只股票分配的资金
            
            # 执行买入订单
            for signal in self.buy_signals:
                size = int(per_stock_cash / signal['price'])  # 计算可买入的股票数量
                if size > 0:
                    self.buy(size, signal['price'], signal['stock_code'])

def run_backtest(strategy, start_date=None, end_date=None, stock_pool='000852.XSHG', data_fields=None):
    """
    运行回测的API函数
    
    参数:
    strategy: 策略类 (MAStrategy 或 ReversalStrategy)
    start_date: 回测起始日期，格式为 'YYYY-MM-DD'
    end_date: 回测结束日期，格式为 'YYYY-MM-DD'
    stock_pool: 股票池代码，默认为中证1000指数 '000852.XSHG'
    data_fields: 需要的数据字段列表，默认为 None (使用所有字段)
    """
    # 设置默认日期
    if start_date is None:
        start_date = datetime(2020,1,2)
    if end_date is None:
        end_date = datetime(2020,12,30)
    
    # 设置默认数据字段
    if data_fields is None:
        data_fields = ['open', 'high', 'low', 'close', 'volume']
        
    # 初始化数据加载器
    data_loader = DataLoader(index=stock_pool)
    final_data = data_loader.process_data()
    
    # 准备数据源
    data_feeds = {}
    for stock_code, stock_data in final_data.groupby(level='stk_id'):
        stock_data = stock_data.reset_index(level='stk_id', drop=True)
        stock_data = stock_data.reset_index(names=['date'])
        
        # 创建数据源
        data_feeds[stock_code] = StockData(
            stock_code=stock_code,
            dataname=stock_data,
            datetime='date',
            open_price='open' if 'open' in data_fields else None,
            high='high' if 'high' in data_fields else None,
            low='low' if 'low' in data_fields else None,
            close='close' if 'close' in data_fields else None,
            volume='volume' if 'volume' in data_fields else None,
            fromdate=start_date,
            todate=end_date
        )
    
    # 设置回测引擎
    brain = Brain()
    brain.adddata(data_feeds)
    brain.addstrategy(strategy)
    
    # 运行回测
    brain.run()
    brain.plot_value()
    
    # 获取基准数据
    benchmark_data = rq.get_price(stock_pool, start_date=start_date, end_date=end_date,
             frequency='1d', fields=None, adjust_type='pre', skip_suspended=False, 
             market='cn', expect_df=True, time_slice=None)['close']
             
    # 计算基准和策略的日收益率
    benchmark_data=benchmark_data.droplevel(0)
    benchmark_returns = benchmark_data.pct_change().dropna()
    strategy_returns = pd.Series(np.diff(brain.broker.daily_values) / brain.broker.daily_values[:-1], 
                               index=brain.broker.daily_dates[1:])

    # 计算累积收益率
    benchmark_cum_returns = (1 + benchmark_returns).cumprod() 
    strategy_cum_returns = (1 + strategy_returns).cumprod() 
    
    # 计算年化超额收益率
    excess_returns = strategy_returns - benchmark_returns
    annual_excess_return = excess_returns.mean() * 252
    
    # 绘制累积收益率对比图
    plt.figure(figsize=(12, 6))
    plt.plot(strategy_cum_returns.index.to_numpy(), strategy_cum_returns.values, label='strategy cum_return')
    plt.plot(benchmark_cum_returns.index.to_numpy(), benchmark_cum_returns.values, label='benchmark cum_return')
    plt.title('Stategy vs Benchmark PnL')
    plt.xlabel('date')
    plt.ylabel('cum_return')
    plt.legend()
    plt.grid(True)
    
    print(f"策略相对基准的年化超额收益率: {annual_excess_return:.2%}")
    plt.show()
    return brain

# 使用示例
# cerebro = run_backtest(MAStrategy, '2022-02-05', '2023-10-30', '000852.XSHG', ['open', 'high', 'low', 'close', 'volume'])