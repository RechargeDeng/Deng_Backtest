<a id="backtesting"></a>

# backtesting

<a id="backtesting.PandasData"></a>

## PandasData Objects

```python
class PandasData()
```

数据源类,用于加载和管理pandas DataFrame数据

<a id="backtesting.PandasData.start"></a>

#### start

```python
def start()
```

数据加载前的准备工作

<a id="backtesting.PandasData.reset"></a>

#### reset

```python
def reset()
```

重置数据容器

<a id="backtesting.PandasData.preload"></a>

#### preload

```python
def preload()
```

预加载所有数据

<a id="backtesting.PandasData.load"></a>

#### load

```python
def load()
```

加载一行数据

<a id="backtesting.PandasData.forward"></a>

#### forward

```python
def forward()
```

为所有数据线添加空位

<a id="backtesting.PandasData.backwards"></a>

#### backwards

```python
def backwards()
```

回退一个数据位置

<a id="backtesting.Order"></a>

## Order Objects

```python
class Order()
```

订单类

<a id="backtesting.Broker"></a>

## Broker Objects

```python
class Broker()
```

经纪商类

<a id="backtesting.Broker.submit"></a>

#### submit

```python
def submit(order)
```

提交订单

<a id="backtesting.Broker.execute"></a>

#### execute

```python
def execute(order, price)
```

执行订单

<a id="backtesting.Broker.next"></a>

#### next

```python
def next(current_date=None, current_prices=None)
```

处理订单队列

<a id="backtesting.Strategy"></a>

## Strategy Objects

```python
class Strategy()
```

策略基类

<a id="backtesting.Strategy.next"></a>

#### next

```python
def next()
```

策略逻辑

<a id="backtesting.Strategy.buy"></a>

#### buy

```python
def buy(size, price=None, stock_code=None)
```

买入

<a id="backtesting.Strategy.sell"></a>

#### sell

```python
def sell(size, price=None, stock_code=None)
```

卖出

<a id="backtesting.Brain"></a>

## Brain Objects

```python
class Brain()
```

回测引擎

<a id="backtesting.Brain.adddata"></a>

#### adddata

```python
def adddata(data)
```

添加数据源

<a id="backtesting.Brain.setbroker"></a>

#### setbroker

```python
def setbroker(broker)
```

设置经纪商

<a id="backtesting.Brain.addstrategy"></a>

#### addstrategy

```python
def addstrategy(strategy_cls, *args, **kwargs)
```

添加策略

<a id="backtesting.Brain.run"></a>

#### run

```python
def run(preload=True, exactbars=False)
```

运行回测

<a id="backtesting.Brain.runstrategies"></a>

#### runstrategies

```python
def runstrategies()
```

运行策略

<a id="backtesting.Brain.plot_value"></a>

#### plot\_value

```python
def plot_value()
```

绘制账户总价值曲线并计算回测指标

<a id="test_api"></a>

# test\_api

<a id="test_api.DataLoader"></a>

## DataLoader Objects

```python
class DataLoader()
```

数据加载类,用于读取和处理双索引DataFrame数据

<a id="test_api.DataLoader.__init__"></a>

#### \_\_init\_\_

```python
def __init__(file_path='模拟cache存储数据.csv', index='000300.XSHG')
```

初始化数据加载器
参数:
    file_path: CSV文件路径

<a id="test_api.DataLoader.load_data"></a>

#### load\_data

```python
def load_data()
```

读取CSV文件并设置双索引

<a id="test_api.DataLoader.generate_stock_pool"></a>

#### generate\_stock\_pool

```python
def generate_stock_pool()
```

生成股票池方法
这里使用示例股票池,实际应用中可根据需求修改选股逻辑

<a id="test_api.DataLoader.filter_data"></a>

#### filter\_data

```python
def filter_data()
```

根据股票池筛选数据

<a id="test_api.DataLoader.process_data"></a>

#### process\_data

```python
def process_data()
```

完整的数据处理流程

<a id="test_api.MAStrategy"></a>

## MAStrategy Objects

```python
class MAStrategy(Strategy)
```

双均线策略

<a id="test_api.MAStrategy.next"></a>

#### next

```python
def next()
```

策略逻辑

<a id="test_api.ReversalStrategy"></a>

## ReversalStrategy Objects

```python
class ReversalStrategy(Strategy)
```

N日反转策略

<a id="test_api.ReversalStrategy.next"></a>

#### next

```python
def next()
```

策略逻辑

<a id="test_api.run_backtest"></a>

#### run\_backtest

```python
def run_backtest(strategy,
                 start_date=None,
                 end_date=None,
                 stock_pool='000852.XSHG',
                 data_fields=None)
```

运行回测的API函数

参数:
strategy: 策略类 (MAStrategy 或 ReversalStrategy)
start_date: 回测起始日期，格式为 'YYYY-MM-DD'
end_date: 回测结束日期，格式为 'YYYY-MM-DD'
stock_pool: 股票池代码，默认为中证1000指数 '000852.XSHG'
data_fields: 需要的数据字段列表，默认为 None (使用所有字段)


