# 京能七站总功率预测服务

接收七站真实功率、温度和湿度，返回未来24小时的 **96点总功率预测**，间隔15分钟，单位MW。当前模型为 `trend_detail_7station_2025_v1`，包含NHITS、PatchTST和StationAttentionHF的TrendDetail融合；无需重新训练。

原预测平台JSON接口：`POST /api/v1/fluxcast/compute`，输入 `point_table + frames`，输出 `result_point`。`varname` 为 `totalPowerForecast`，`event_key` 沿用 `JNH.Fluxcast.Compute`。历史/天气未就绪返回HTTP 200和空结果，附原因。

独立实测接口 `POST /api/v1/fluxcast/observations`：接收七站19个功率测点的单帧UTC数据，返回各站出力、占比、数据状态、实测总功率及可匹配的预测偏差。与预测服务在同一容器内分进程运行，默认端口分别为8000、8002，缓存分开。模型权重和融合算法不变；功率输入按逐测点负值归零的新规则处理。

**时间规则：历史训练数据已确认为北京时间（`Asia/Shanghai`）。接口输入、输出均为UTC；预测模型内部转换为北京时间，保持与训练特征一致。输出保留本次请求的时间格式，预测请求的96帧须使用一致格式。实测接口按UTC同刻匹配，不重复换算。模型权重不变；目标服务器仍需联调验收，使用与当前源码提交对应的镜像。**

## 交接文档

| 内容 | 文件 |
|---|---|
| 预测请求、响应、缺测、原因码与完整JSON样例 | [接口交接说明](docs/接口交接说明.md) |
| 预测所需七站35个原始测点编码及单位 | [测点需求清单](docs/测点需求清单.md) |
| 构建、启动、缓存、镜像交付与验收 | [Docker部署运行说明](docs/Docker部署运行说明.md) |
| 单帧实测接收、缺测状态、UTC匹配及输入输出样例 | [实测接口说明](docs/实测接口说明.md) |
| 每个返回变量的名称、类型、单位、时间和返回条件 | [返回字段说明](docs/返回字段说明.md) |

源码交付至公司指定的功率预测GitLab分支，包含运行代码、权重、文档、样例、测试和部署配置。大型镜像及SHA256校验文件交付到公司制品库或指定目录，不提交普通源码历史。本工程未配置公司GitLab Runner流水线。

## 文件用途

| 路径 | 用途 |
|---|---|
| `app/api.py`、`app/platform_adapter.py` | HTTP路由、平台输入输出适配、空结果和日志 |
| `app/time_policy.py` | UTC与模型北京时间转换、缓存时间规则版本 |
| `app/input_adapter.py`、`app/history_cache.py` | 测点、特征、缺失处理及历史缓存 |
| `app/predict.py`、`app/models/` | 预测计算代码 |
| `observations_service/` | 新实测服务、独立归档、同容器启动器及该模块文档和测试 |
| `models/active_model.json`、`models/versions/` | 当前模型配置与训练权重，必须保留 |
| `tests/` | 接口、缓存、打包与真实数据滚动测试 |
| `docs/`、`examples/` | 交接文档和JSON样例 |
| `Dockerfile`、`requirements.txt`、`compose.yaml`、`.env.example` | 镜像构建和部署配置 |
| `.github/workflows/build-image.yml` | 平台接口、重建恢复、73天评分及镜像导入测试 |

`runtime/` 是预测运行缓存，`runtime_observations/` 是实测与预测曲线归档，均不交付、不预装。`tests/`、`docs/`、`examples/` 及实测目录中的文档、测试、样例不进入运行镜像。

## 数据接入

### 预测接口（8000）

1. 每次提交七站同一时间范围的完整快照，共96帧，每15分钟一帧；预测调用频率沿用原流程，可每天生成一次未来24小时曲线。
2. `point_table` 列全19个功率、8个温度、8个湿度编码，`frames` 内直接使用测点编码作为key。缺测推荐写 `null`，也兼容省略该帧测点。
3. 预测服务先从早到晚补历史，或持续累计。672个连续且天气可构造的点（7天）满足后才预测；不足返回200和空结果，不清空缓存。
4. 筛选 `result_point` 中 `varname=totalPowerForecast` 的96条记录，读取未来七站总功率。另有点数、间隔统计和 `extra_info` 中的批次、生成时间、预测范围。`reason` 和 `message` 解释空结果。HTTP 200不等于已有预测。

预测输入中的功率缺失补0，有限负功率逐测点归零，真实0有效；天气仅沿用该测点过去真实值。平台不传未来天气，不提供模型内部特征，也不需要把训练CSV上传生产服务器。升级到本功率规则须使用新的预测runtime目录并补传至少7天连续可用历史，旧目录保留备份；实测归档可保留并自动升级。

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/fluxcast/compute" -H "Content-Type: application/json" --data-binary @examples/platform_input_example.json
curl "http://127.0.0.1:8000/api/v1/fluxcast/compute/latest"
```

Windows使用 `curl.exe`；跨机器地址由部署方提供，Compose默认仅监听本机。预测样例由实际CSV的北京时间换算为UTC；空缓存只提交一天预测样例不会产生预测。平台已经发送UTC时，无需再手动加减8小时。

### 实测接口（8002）

平台每15分钟推送一次七站同一时刻的19个功率测点，使用 `point_table + frames`，每次恰好1帧；不传温湿度，无需7天历史预热。

19点完整且合计可计算才返回实测总功率，有符合条件的同刻UTC预测且差值可计算才返回“预测减实测”的偏差。有限负功率逐测点归零，缺测不补0。缺测或计算溢出只省略受影响的数值，其他可计算站出力、各站计数和状态照常返回；`extra_info` 的 `reason`、`message` 说明具体原因及未返回字段，并记录日志。实测请求不触发预测。

对接用JSON位于根目录 `examples/`：见 [实测输入](examples/observations_input_example.json)、[实测输出](examples/observations_output_example.json)。无匹配预测及缺测样例、字段和返回条件见 [实测接口说明](docs/实测接口说明.md)。

```bash
curl -X POST "http://127.0.0.1:8002/api/v1/fluxcast/observations" -H "Content-Type: application/json" --data-binary @examples/observations_input_example.json
```

## 构建与启动

在同架构构建机安装Docker Engine和Compose 2.20+，在源码根目录执行：

```bash
docker build -t jingneng-power-forecast:7station-platform-v1 .
```

首次复制 `.env.example` 为 `.env`，填写 `POWER_FORECAST_IMAGE=jingneng-power-forecast:7station-platform-v1`，已有配置不要覆盖。首次联调使用新的专用runtime目录，不复制旧接口缓存。

```bash
docker compose config --quiet
docker compose up -d --pull never --wait --wait-timeout 300
docker compose ps
docker compose logs --tail 100 forecast
```

构建打包现有权重，不重新训练。源码构建需要下载基础镜像和依赖；离线镜像导入、持久化、端口和运维方法见部署说明。服务没有内置鉴权和HTTPS，交由网关或受控内网管理。

## 版本与下载

本仓库是2026-10-08独立交付版本，包含96点预测、实测与偏差接口及异常值处理。模型权重仍为同一套七站模型，输出不含固定历史参考 `accuracy`。仓库不包含其他项目或旧仓库的提交历史。

源码仓库：[jingneng-power-forecast-observations-20261008](https://github.com/zhangqian-1/jingneng-power-forecast-observations-20261008)。[下载当前main源码](https://github.com/zhangqian-1/jingneng-power-forecast-observations-20261008/archive/refs/heads/main.zip)；公司GitLab交接建议使用Release中与镜像同提交的源码ZIP。源码包括模型、文档、样例及用于构建验证的真实测试CSV；运行镜像不带测试集。

已封装文件下载：[本仓库Releases](https://github.com/zhangqian-1/jingneng-power-forecast-observations-20261008/releases)。构建、接口、重启、73天滚动评分及镜像导出导入检查通过后，自动发布以下附件：

| 文件 | 用途 |
|---|---|
| `source-提交号.zip` 和对应 `.sha256` | 同提交源码、模型、交接文档和样例，用于公司GitLab交接 |
| `offline-image-amd64-…zip` 和对应 `.sha256` | Linux AMD64离线镜像及Compose启动配置，可直接交给部署人员 |
| `container-checks-amd64-…zip` | 构建和接口验证记录、实际滚动评分，供核验使用 |

本次仅提供 `linux/amd64` 镜像，对应服务器 `x86_64`，不适用于ARM服务器。交付时核对源码、镜像、`release.json` 和 `SHA256SUMS` 对应关系。运行缓存、历史结果和开发环境不交付。构建进度见 [Actions](https://github.com/zhangqian-1/jingneng-power-forecast-observations-20261008/actions)；只有Release附件上传成功才表示封装交付完成。正式上线前仍须完成目标服务器平台联调。
