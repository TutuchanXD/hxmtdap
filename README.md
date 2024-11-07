<h1 id="iQ5VT">简介</h1>

HXMT数据处理Pipeline (hxmtdap)项目旨在以更方便的方式调用HXMTDAS的命令，来产生标准化的慧眼卫星数据产品。

<h1 id="h3n8H">安装与依赖</h1>

使用以下命令安装hxmtdap：

```bash
pip install hxmtdap-0.2.3-py3-none-any.whl
```

该项目依赖以下Python库：

```shell
astropy>=4.0.0
cycler>=0.10.0
hvplot>=0.10.0
ipython>=8.10.0
matplotlib>=3.5.0
networkx>=3.1.0
numpy>=1.25.0
pandas>=2.0.0
panel>=1.2.0
pyinterval>=1.2.0
rich>=12.0.0
scipy>=1.10.0
stingray>=2.0
uncertainties>=3.2.0
```

<h1 id="ERbVO">快速开始</h1>

以下示例展示如何从零开始快速生成一个源的谱、光变曲线和功率谱：

```python
from hxmtdap import LEDA, MEDA, HEDA

# LE 探测器

# 初始化
lep = LEDA("<your_expid_path>", "DEBUG")
lep.lepical()
lep.lerecon()
lep.legtigen()
lep.legticorr()
lep.lescreen()

# 生成光变曲线和功率谱
lep.lelcgen(minE=1, maxE=10)
lep.lebkgmap_lc()
lep.lenetlcgen()
lep.lenetpdsgen(segment=256., rebin=-0.06)
lep.lermspdsgen(segment=256., rebin=-0.06)

# 生成谱和谱背景、响应文件
lep.lespecgen()
lep.lerspgen()
lep.lebkgmap_spec()
lep.grppha_le()

lep.close()

```

```python
# ME 探测器

# 初始化
mep = MEDA("<your_expid_path>", "DEBUG")
mep.mepical()
mep.megrade()
mep.megtigen()
mep.megticorr()
mep.mescreen()

# 生成光变曲线和功率谱
mep.melcgen(minE=10, maxE=35)
mep.mebkgmap_lc()
mep.menetlcgen()
mep.menetpdsgen(segment=256., rebin=-0.06)
mep.mermspdsgen(segment=256., rebin=-0.06)

# 生成谱和谱背景、响应文件
mep.mespecgen()
mep.merspgen()
mep.mebkgmap_spec()
mep.grppha_me()

mep.close()
```

```python
# HE 探测器

# 初始化
hep = HEDA("<your_expid_path>", "DEBUG")
hep.hepical()
hep.hegtigen()
hep.hescreen()

# 生成光变曲线和功率谱
hep.helcgen(minE=35, maxE=150)
hep.hebkgmap_lc()
hep.henetlcgen()
hep.henetpdsgen(segment=256., rebin=-0.06)
hep.hermspdsgen(segment=256., rebin=-0.06)

# 生成谱和谱背景、响应文件
hep.hespecgen()
hep.herspgen()
hep.hebkgmap_spec()
hep.grppha_he()

hep.close()
```

<h1 id="ySKj7">特点与功能</h1>
<h2 id="EdvuB">解耦数据处理</h2>

LE、ME和HE探测器的pipeline对应LEDA、MEDA和HEDA三个不同的模块，彼此之间没有交叉关系，可以根据所需的能段来选择使用不同的探测器模块来生成数据产品

<h2 id="nadW5">日志记录</h2>

hxmtdap可以保留完整的数据处理日志，可以在控制台输出数据处理过程且保留日志文件，日志文件的默认路径为`<your_expid_path>/log`；在初始化时，可以传入`logger_level`参数调整日志级别

```python
from hxmtdap import LEDA

lep = LEDA("<your_expid_path>", logger_level="DEBUG") # 推荐日志级别为DEBUG
```

<h2 id="bhQEX">数据持久化</h2>

hxmtdap每次成功生成数据文件后，会将其以`node`的方式记录在图中（图文件默认保存在`<your_expid_path>/Graph`）；每次初始化都会尝试载入图文件，以获取保存的数据处理状态

```python
from hxmtdap import LEDA

lep = LEDA("<your_expid_path>")
lep.lepical()
lep.lerecon()
lep.legtigen()
lep.legticorr()
lep.lescreen()

```

```python
from hxmtdap import LEDA

lep = LEDA("<your_expid_path>")

lep.lespecgen() # 无需运行前置命令，直接读取图中的node以获取之前的数据处理进度
```

<h2 id="WCEJR">数据文件可视化</h2>

hxmtdap使用`node`和图来记录数据文件的状态、路径与参数、数据文件之间的血缘关系；可以通过`show`方法来查看当前数据处理状态下的数据文件图

```python
lep = LEDA("<your_expid_path>")
lep.show()
```



![](https://cdn.nlark.com/yuque/0/2024/png/2907884/1719823712796-a28aa133-ded5-419f-836c-9fc057b55b69.png)

图上的每个`node`都表示一个数据文件，紫色表示1L级数据文件，蓝色表示校正(Calibration)数据文件，绿色表示筛选(Screen)数据文件，黄色表示数据产品(High Level Product)，红色表示扩展数据产品(Ex)

鼠标放置在`node`上面时会高亮显示其信息；`node`之间的连线表示数据之间的引用关系；例如图上的`LE_pi`在前一层有`LE_Evt`和`LE_TH`的连线，表示生成`LE_pi`时传入了这两个数据。

如果希望改变`node`的名称，只需要在生成数据时显式的传入参数`node`

```python
from hxmtdap import LEDA

lep = LEDA("<your_expid_path>")

lep.lespecgen(node='myspec')
```

<h2 id="XJP2W">显式调用数据文件</h2>

可以通过`recoder`模块调用已生成的数据文件

```python
from hxmtdap import LEDA

lep = LEDA("<your_expid_path>")
lep.lespecgen(node='myspec')

lep.recoder.myspec # 从这里获取刚生成的文件完整路径
```

<h2 id="XNPcQ">自定义参数</h2>

hxmtdap在未指定参数时，会通过读取准备好的`json`默认参数文件（在包的`json`目录）获取默认参数；如果希望在执行某命令时使用自定义参数，只需要显式的传入参数值

```python
from hxmtdap import LEDA

lep = LEDA("<your_expid_path>")

lep.lelcgen(starttime=369597279, stoptime=369598059)
```

<h2 id="KPZdZ">动态数据结构</h2>

hxmtdap通过`status`子模块检测活动的节点，当前的数据文件生成，仅依赖最新的前置数据文件，比如

```python
from hxmtdap import LEDA

lep = LEDA("<your_expid_path>")

lep.lelcgen(minE=1, maxE=2)
lep.lebkgmap_lc() # 这里生成的光变曲线背景就是上面1-2keV的，无需在指定输入的光变曲线

lep.lelcgen(minE=2, maxE=4)
lep.lebkgmap_lc() # 这里生成的光变曲线背景是2-4keV的
```

<h2 id="GRLoC">能谱与背景的曝光时间对齐</h2>

`lebkgmap`、`mebkgmap`和`hebkgmap`在生成谱背景文件时，只会通过读取`gti`文件来获取背景的曝光时间；如果在生成能谱阶段显式的设置了`starttime`和`stoptime`，则后续生成的能谱背景的曝光时间无法与谱文件对齐。hxmtdap通过假`gti`文件的方式来保证谱背景和能谱的对齐

```python
from hxmtdap import LEDA

lep = LEDA("<your_expid_path>")

lep.lespecgen(starttime=369597279, stoptime=369603699)
lep.lebkgmap_spec() # 这里生成的背景文件曝光时间是与谱对齐的
```

<h2 id="h8srI">更规则的命名</h2>

hxmtdap使用更规则清晰的命名规则

```python
from hxmtdap import HEDA

hep = HEDA("<your_expid_path>")

hep.helcgen(minE=35, maxE=46, binsize=1/2048)
# 生成的文件在 <your_expid_path>/lc/HE/raw/<expid>_HE_raw_35-46keV_488usBin.lc

hep.helcgen(minE=35, maxE=46, binsize=1/512)
# 生成的文件在 <your_expid_path>/lc/HE/raw/<expid>_HE_raw_35-46keV_2msBin.lc

hep.hebkgmap_lc()
# 生成的文件在 <your_expid_path>/lc/HE/bkg/<expid>_HE_lcbkg_35-46keV_2msBin.lc

hep.henetlcgen()
# 生成的文件在 <your_expid_path>/lc/HE/net/<expid>_HE_lcnet_35-46keV_2msBin.lc

hep.hentpdsgen(segment=256., rebin=-0.06)
# 生成的文件在 <your_expid_path>/pds/HE/leahy/-0.06/<expid>_HE_256s_leahy_35-46keV_2msBin.fps
```

<h2 id="Yigzf">允许在IPython或Jupyter环境使用</h2>

hxmtdap可以检测当前运行环境，来配置相应的环境变量

<h2 id="DyiHu">光变曲线相关</h2>
<h3 id="acf6P">显式的输入能段来生成光变曲线</h3>

hxmtdap内置pi转换，可以直接显式的输入

```python
from hxmtdap import LEDA

lep = LEDA("<your_expid_path>")

lep.lelcgen(minE=1, maxE=2) # 不需要输入minPI maxPI
```

<h3 id="PcuGA">光变曲线、背景和功率谱自动绘图</h3>

对于光变曲线可以自动绘制`gti`范围内的断轴图像，图像路径与光变曲线相同（默认bin为1秒）

![](https://cdn.nlark.com/yuque/0/2024/png/2907884/1719826471617-196a7c17-a130-4c67-a6f3-d202787b3cbb.png)

![](https://cdn.nlark.com/yuque/0/2024/png/2907884/1719826505341-15308fa4-7526-418f-88d9-131a5a4a14e6.png)

<h2 id="rCTSg">对并行的支持</h2>

经过测试，hxmtdap支持多进程并行；在并行时单进程可能占用200-1600M活动内存(会回收)，需要控制最大进程数以防止内存溢出；以下是一个使用进程池的例子：

```python
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from natsort import natsorted

from hxmtdap import LEDA, MEDA, HEDA
from hxmtdap.core.logger import setup_logger
from hxmtdap.tools.utils import get_expID

MAXWORKERS = 50

# 曝光数据路径组成的列表
data_lst = [...]

global_logger = setup_logger("all.log", <your_log_path>)

le_lcrange = ["1-2", "2-5", "5-10", "1-10"]
me_lcrange = ["10-16", "16-35", "10-35"]
he_lcrange = ["35-46", "46-60", "60-76", "76-100", "100-200", "35-200"]

rebin_range = [0, -0.02, -0.04, -0.06, -0.08, -0.1]


def process_lep(i):
    try:
        lep = LEDA(i, logger_level="DEBUG")
        lep.logger.handlers[1].setLevel(logging.ERROR)
        lep.lepical()
        lep.lerecon()
        lep.legtigen()
        lep.legticorr()
        lep.lescreen()

        for lcrange in le_lcrange:
            minE, maxE = lcrange.split("-")
            lep.lelcgen(minE=minE, maxE=maxE, node=f"LE_lc_{minE}_{maxE}keV")
            lep.lebkgmap_lc(node=f"LE_lcbkg_{minE}_{maxE}keV")
            lep.lenetlcgen(node=f"LE_lcnet_{minE}_{maxE}keV")
            for rebin in rebin_range:
                lep.lenetpdsgen(
                    segment=256, rebin=rebin, node=f"LE_pdsnet_{minE}_{maxE}keV"
                )
                lep.lermspdsgen(
                    segment=256, rebin=rebin, node=f"LE_pdsrms_{minE}_{maxE}keV"
                )

        lep.lespecgen()
        lep.lebkgmap_spec()
        lep.lerspgen()
        lep.grppha_le()
        lep.close()
        global_logger.info(f"Finished LEP {i}.")
    except Exception as e:
        global_logger.error(f"Error processing LEP for input {get_expID(i)}: {e}")


def process_mep(i):
    try:
        mep = MEDA(i, logger_level="DEBUG")
        mep.logger.handlers[1].setLevel(logging.ERROR)
        mep.mepical()
        mep.megrade()
        mep.megtigen()
        mep.megticorr()
        mep.mescreen()

        for lcrange in me_lcrange:
            minE, maxE = lcrange.split("-")
            mep.melcgen(minE=minE, maxE=maxE, node=f"ME_lc_{minE}_{maxE}keV")
            mep.mebkgmap_lc(node=f"ME_lcbkg_{minE}_{maxE}keV")
            mep.menetlcgen(node=f"ME_lcnet_{minE}_{maxE}keV")
            for rebin in rebin_range:
                mep.menetpdsgen(
                    segment=256, rebin=rebin, node=f"ME_pdsnet_{minE}_{maxE}keV"
                )
                mep.mermspdsgen(
                    segment=256, rebin=rebin, node=f"ME_pdsrms_{minE}_{maxE}keV"
                )

        mep.mespecgen()
        mep.mebkgmap_spec()
        mep.merspgen()
        mep.grppha_me()
        mep.close()
        global_logger.info(f"Finished MEP {i}.")
    except Exception as e:
        global_logger.error(f"Error processing MEP for input {get_expID(i)}: {e}")


def process_hep(i):
    try:
        hep = HEDA(i, logger_level="DEBUG")
        hep.logger.handlers[1].setLevel(logging.ERROR)
        hep.hepical()
        hep.hegtigen()
        hep.hescreen()

        for lcrange in he_lcrange:
            minE, maxE = lcrange.split("-")
            hep.helcgen(minE=minE, maxE=maxE, node=f"HE_lc_{minE}_{maxE}keV")
            hep.hebkgmap_lc(node=f"HE_lcbkg_{minE}_{maxE}keV")
            hep.henetlcgen(node=f"HE_lcnet_{minE}_{maxE}keV")
            for rebin in rebin_range:
                hep.henetpdsgen(
                    segment=256, rebin=rebin, node=f"HE_pdsnet_{minE}_{maxE}keV"
                )
                hep.hermspdsgen(
                    segment=256, rebin=rebin, node=f"HE_pdsrms_{minE}_{maxE}keV"
                )

        hep.hespecgen()
        hep.hebkgmap_spec()
        hep.herspgen()
        hep.grppha_he()
        hep.close()
        global_logger.info(f"Finished HEP {i}.")
    except Exception as e:
        global_logger.error(f"Error processing HEP for input {get_expID(i)}: {e}")


def main(dl):
    with ProcessPoolExecutor(max_workers=MAXWORKERS) as executor:
        futures = []
        for expidpath in dl:
            futures.append(executor.submit(process_lep, path))
            futures.append(executor.submit(process_mep, path))
            futures.append(executor.submit(process_hep, path))
        for future in as_completed(futures):  # 所有任务在执行时，在此阻塞
            try:
                future.result(timeout=900)
            except TimeoutError as e:
                logging.error(f"Task timeout with exception: {e}")
            except Exception as e:
                logging.error(f"Task failed with exception: {e}")
            finally:
                del future

main(data_lst)
```

<h2 id="GS6ap">工具函数</h2>

hxmtdap提供了一些对数据文件读取和操作的工具，分别有：

+ `hxmtdap.tools.utils` 通用
+ `hxmtdap.tools.lcutils` 光变曲线
+ `hxmtdap.tools.pdsutils` 功率谱
+ `hxmtdap.tools.xutils` 对xspec结果的解析

其中`hxmtdap.tools.xutils.LogDataResolver`可以解析xspec拟合过程生成的log文件，提取最佳拟合参数：

```python
from hxmtdap.tools.xutils import LogDataResolver

log_resolver = LogDataResolver(<your_logfile_path>)
log_resolver.parameters_stat # 返回由参数信息组成的字典
```

这个类还提供了一个`reconstruct_model`方法来重建基于原模型的一个子模型xcm格式字符串，可以帮助你分解最佳拟合模型中的各个子模型结果，一个例子：

```python
from tempfile import NamedTemporaryFile
import xspec

# 如果你的模型是 constant*TBabs(diskbb + relxill + guassian)
# 总有这样的子模型组合：
#	constant*TBabs*diskbb + constant*TBabs*relxill + constant*TBabs*gaussian

xspec.Xset.restore(<your_original_xcmfile>)
xspec.Xset.openLog('logfile.log')
xspec.AllData.show()
xspec.AllModels.show()
xspec.Fit.show()
xspec.Xset.closeLog()

log_resolver = LogDataResolver('logfile.log')

# 返回这个子模型的xcm字符串，参数与原本一致
xcmstring = log_resolver.reconstruct_model('constant*TBabs*diskbb') 

with NamedTemporaryFile(mode="w+t", delete=True) as f:
    f.write(xcmstring)
    f.write('\n')
    f.flush()
    xspec.Xset.restore(f.name)
    
... # 之后可以绘图并提取模型数据

```

如果你的模型中有多个重名组件，程序的规则是需要在重建的组件下添加**组件序号角标**，一个例子比如：

```python
# 你的模型为 powerlaw + lorentz + lorentz + lorentz
# 第一个组件是 powerlaw，第二个是lorentz或lorentz_2，第三个是lorentz_3，第四个是lorentz_4

log_resolver = LogDataResolver('logfile.log')

# 第一个组件的xcm字符串
log_resolver.reconstruct_model('powerlaw')

# 第二个组件的xcm字符串
log_resolver.reconstruct_model('lorentz')

# 第三个组件的xcm字符串
log_resolver.reconstruct_model('lorentz_3')

# 第四个组件的xcm字符串
log_resolver.reconstruct_model('lorentz_4')

# 当然你可以自由组合,但组件顺序要和原来一致
log_resolver.reconstruct_model('lorentz + lorentz_4')
```

<h1 id="Pv0oy">如何贡献</h1>

<font style="color:rgb(13, 13, 13);">鼓励其他开发者参与进来，并提供详细的说明如何贡献，可以包括：</font>

+ <font style="color:rgb(13, 13, 13);">报告 bug</font>
+ <font style="color:rgb(13, 13, 13);">提交功能请求</font>
+ <font style="color:rgb(13, 13, 13);">提交合并请求</font>

<h1 id="J72XP">许可证</h1>

当前版本为内部测试版，所有代码和文档的版权均由**Chenxu Gao**所有。未经许可，不得分发此软件及其文档。

计划测试阶段结束后使用MIT开源许可。





