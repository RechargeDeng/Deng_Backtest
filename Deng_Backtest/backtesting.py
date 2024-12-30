import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from itertools import groupby
from matplotlib.font_manager import FontProperties
from datetime import datetime
from collections import deque, OrderedDict

class PandasData:
    """数据源类,用于加载和管理pandas DataFrame数据"""
    """该类将已经转换为pandas Dataframe的数据制作成数据线的形式，
    该类的核心方法load能够使得该类数据以流的方法传入回测系统，以避免未来信息的使用"""
    
    def __init__(self, dataname,
                 datetime=None,
                 open_price=None,
                 high=None,
                 low=None,
                 close=None,
                 volume=None,
                 openinterest=None,
                 fromdate=None,
                 todate=None):
        
        self.dataname = dataname
        self.datetime = datetime
        self.open_price = open_price
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.openinterest = openinterest
        self._colmapping = OrderedDict()
        self.start_load=False
        self.stop_load=False
        # 设置列映射
        for name, col in ((name, getattr(self, name)) 
                         for name in ['datetime', 'open_price', 'high', 'low', 
                                    'close', 'volume', 'openinterest']):
            if col is not None:
                self._colmapping[name] = col
        #print(self._colmapping)
        # 数据容器
        self._idx = 0  # 当前读取的DataFrame行数
        self.lines = {name: deque() for name in self._colmapping.keys()}
        
        # 设置回测区间
        self.fromdate = fromdate
        self.todate = todate
        
        # 数据状态
        self._started = False
        
    def start(self):
        """数据加载前的准备工作"""
        self._idx = 0
        self._started = True
        
    def _start_finish(self):
        """时区时间相关参数的最终调整"""
        pass  # 简化版本不处理时区
        
    def _start(self):
        """启动数据加载流程"""
        self.start()
        if not self._started:
            self._start_finish()
            
    def reset(self):
        """重置数据容器"""
        self._idx = 0
        for line in self.lines.values():
            line.clear()
            
    def preload(self):
        """预加载所有数据"""
        while self.load():
            pass
        print(self.lines)
            
    def load(self):
        """加载一行数据"""
        if self._idx >= len(self.dataname):
            return False
            
        # 读取当前行
        row = self.dataname.iloc[self._idx]
        #print(row)
        # 检查日期是否在回测区间内
        dt = row[self._colmapping['datetime']]
        #print(dt)
        if self.fromdate and dt < self.fromdate:
            self._idx += 1
            return True
        if self.todate and dt > self.todate:
            self._idx += 2
            self.stop_load=True
            return False
        if self.fromdate and dt >= self.fromdate:
            self.start_load=True
        # 添加数据到容器
        self.forward()
        for name, col in self._colmapping.items():
            self.lines[name][-1] = row[col]
        #print(self.lines)    
        self._idx += 1
        return True
        
    def forward(self):
        """为所有数据线添加空位"""
        for line in self.lines.values():
            line.append(None)
            
    def backwards(self):
        """回退一个数据位置"""
        for line in self.lines.values():
            if line:
                line.pop()


class Order:
    """订单类"""
    Created, Submitted, Accepted, Partial, Completed, \
        Canceled, Expired, Margin, Rejected = range(9)
        
    def __init__(self, size, price=None, exectype=None, stock_code=None):
        self.size = size
        self.price = price
        self.exectype = exectype
        self.status = self.Created
        self.stock_code = stock_code
        

class Broker:
    """经纪商类"""
    def __init__(self, cash=10000000.0):
        self.cash = cash
        self.positions = {}  # 不同股票的持仓 {stock_code: position}
        self.value = 0
        
        self.submitted = []  # 提交订单队列
        self.pending = []    # 等待执行队列
        
        self.daily_values = []  # 记录每日账户总价值
        self.daily_dates = []   # 记录对应日期
        
    def submit(self, order):
        """提交订单"""
        order.status = Order.Submitted
        self.submitted.append(order)
        return order
        
    def execute(self, order, price):
        """执行订单"""
        cost = order.size * price
        if cost > self.cash and order.size > 0:  # 买入检查资金
            order.status = Order.Rejected
            return False

        # 卖出检查持仓
        if order.size < 0:  # 卖出订单
            if order.stock_code not in self.positions or \
               abs(order.size) > self.positions[order.stock_code]+2:
                order.status = Order.Rejected
                return False
            
        # 更新对应股票的持仓
        if order.stock_code not in self.positions:
            self.positions[order.stock_code] = 0
        self.positions[order.stock_code] += order.size
        
        self.cash -= cost
        # 计算所有股票的总市值
        """
        total_stock_value = sum(pos * price for pos in self.positions.values())
        self.value = self.cash + total_stock_value
        """
        order.status = Order.Completed
        return True
        
    def next(self, current_date=None, current_prices=None):
        """处理订单队列"""
        # 处理提交订单
        #print(len(self.submitted))
        for order in self.submitted:
            self.pending.append(order)
        self.submitted = []
        
        # 处理等待订单
        for order in self.pending:
            if order.status not in [Order.Submitted, Order.Accepted]:
                continue
            self.execute(order, order.price)
        self.pending = []
        # 记录每日账户总价值
        if current_date and current_prices:
            total_stock_value = sum(pos * current_prices.get(code, 0) 
                                  for code, pos in self.positions.items())
            self.value = self.cash + total_stock_value
            self.daily_dates.append(current_date)
            self.daily_values.append(self.value)


class Strategy:
    """策略基类"""
    def __init__(self, broker, data):
        self.broker = None #broker
        self.data = data
        self.position = 0
        
    def next(self):
        """策略逻辑"""
        pass
        
    def buy(self, size, price=None, stock_code=None):
        """买入"""
        return self.broker.submit(Order(size, price, stock_code=stock_code))
        
    def sell(self, size, price=None, stock_code=None):
        """卖出"""
        return self.broker.submit(Order(-size, price, stock_code=stock_code))

class Brain:
    """回测引擎"""
    def __init__(self):
        self.datas = {}
        self.broker = None
        self.strategy = None
        
    def adddata(self, data):
        """添加数据源"""
        self.datas=data
        
    def setbroker(self, broker):
        """设置经纪商"""
        self.broker = broker
        
    def addstrategy(self, strategy_cls, *args, **kwargs):
        """添加策略"""
        self.strategy = strategy_cls
        self.stratargs = args
        self.stratkwargs = kwargs
        
    def run(self, preload=True, exactbars=False):
        """运行回测"""
        return self.runstrategies()
        
    def runstrategies(self):
        """运行策略"""
        # 启动经纪商
        if not self.broker:
            self.broker = Broker()
        
        # 加载数据
        for data in self.datas.values():
            data.reset()
            data._start()      
        
        # 创建策略实例
        st = self.strategy(broker=self.broker, data=None, 
                          *self.stratargs, **self.stratkwargs)
        
        # 回测主循环
        while data.stop_load!=True:
            # 加载数据
            current_prices = {}
            current_date = None
            for stock_code, data in self.datas.items():
                data.load()
                if len(data.lines['datetime']) > 0:
                    current_date = data.lines['datetime'][-1]
                    #print(current_date)
                    current_prices[stock_code] = data.lines['close'][-1]
            #print(len(data.lines['datetime']))
            if data.start_load!=True:
                continue
            
            # 创建策略实例
            st.data = self.datas 
                              
            # 执行订单,传入当前日期和价格信息
            self.broker.next(current_date, current_prices)
            
            # 运行策略
            st.next()
            
            # 每日交易完成后的信息展示
            print('='*50)
            print(f'📅 {current_date.strftime("%Y-%m-%d")} 交易日交易顺利完成')
            print(f'💰 当前账户总资产: ¥{self.broker.daily_values[-1]:,.2f}')
            print('='*50)
            print()
            
        return True
    def plot_value(self):
        """绘制账户总价值曲线并计算回测指标"""
        import matplotlib.pyplot as plt
        import numpy as np
        from datetime import datetime
        
        # 绘制账户价值曲线
        plt.figure(figsize=(12, 6))
        plt.plot(self.broker.daily_dates, self.broker.daily_values)
        plt.title('Account Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.grid(True)
        
        # 计算回测指标
        returns = np.diff(self.broker.daily_values) / self.broker.daily_values[:-1]
        
        # 计算年化收益率
        total_days = (self.broker.daily_dates[-1] - self.broker.daily_dates[0]).days
        annual_return = (self.broker.daily_values[-1] / self.broker.daily_values[0]) ** (365/total_days) - 1
        
        # 计算年化波动率
        daily_std = np.std(returns)
        annual_volatility = daily_std * np.sqrt(252)
        
        # 计算夏普比率 (假设无风险利率为3%)
        risk_free_rate = 0.03
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        
        # 计算最大回撤率
        max_drawdown = 0
        peak = self.broker.daily_values[0]
        for value in self.broker.daily_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # 打印回测指标
        print(f"年化收益率: {annual_return:.2%}")
        print(f"年化波动率: {annual_volatility:.2%}")
        print(f"夏普比率: {sharpe_ratio:.2f}")
        print(f"最大回撤率: {max_drawdown:.2%}")
        
        plt.show()

        