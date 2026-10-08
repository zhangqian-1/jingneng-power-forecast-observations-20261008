# 实测功率与预测偏差接口

本模块与96点预测服务放在同一个容器，分别运行、分别保存数据。模型权重、融合算法及UTC与北京时间转换不变。功率输入按逐测点负值归零规则处理，实测请求不会触发预测。后端变量字典见 [返回字段说明](../docs/返回字段说明.md)。

## 1. 如何调用

| 用途 | 请求 | 容器端口 |
|---|---|---|
| 原预测：过去7天预测未来1天 | `POST /api/v1/fluxcast/compute` | 8000 |
| 原预测：查询最近成功的96点曲线 | `GET /api/v1/fluxcast/compute/latest` | 8000 |
| 新实测：接收一个时刻的测点，返回总功率和可计算的偏差 | `POST /api/v1/fluxcast/observations` | 8002 |
| 新实测：服务健康状态 | `GET /health` | 8002 |

预测仍按原流程调用，可每天生成一次未来24小时曲线；本模块不会定时生成预测。公司每15分钟主动POST一次实测，不是算法主动向公司拉数据。不使用 `source_id`，每个请求包含七站同一时刻的数据。

在生产部署包根目录调用：

```bash
curl -X POST "http://127.0.0.1:8002/api/v1/fluxcast/observations" -H "Content-Type: application/json" --data-binary @observations_service/examples/input_complete.json
```

Windows使用 `curl.exe`。端口可以通过部署配置改为宿主机其他端口，网关也可以把两个路径映射到同一外部地址。服务无内置鉴权和HTTPS，必须由网关或内网访问控制保护。

## 2. 输入

完整可发送文件：[input_complete.json](examples/input_complete.json)。

- 顶层只有 `point_table` 和 `frames`；`frames` **恰好1帧**，不是预测接口的96帧。
- `point_table` 列全19个功率测点编码，不重复；frame内直接以测点编码为key，保留原编码中的点号和冒号。不传温湿度、不传预先求和的总值。
- 数值单位MW。真实0有效；有限负功率逐测点归零后再求和。字符串、布尔值、null及其他不可用值视为该测点缺测，不把字符串自动当数值。
- 缺测推荐填 `null`，也兼容省略frame中的该字段，但 `point_table` 仍应列全。未知编码、缺少表内编码、帧数错误属于请求格式错误。
- `timestamp` 是该点对应的UTC时刻，不是到达服务器的时间。接受 `2026-11-25 05:00:00`、`2026-11-25T05:00:00Z`、UTC后缀 `+00:00`/`+0000` 等原预测支持的格式；无后缀也按UTC。小数秒可有1至9位，但必须全为0。
- 必须是00/15/30/45分、0秒，不做取整。拒绝非零时区偏移；输出原样保留本帧时间写法。新模块**不再加减8小时**，直接与原预测已返回的UTC目标时刻匹配。
- 请求头 `Content-Type: application/json`，使用UTF-8 JSON及Content-Length，不支持分块传输。最大256 KiB。NaN、Infinity不是合法JSON，请用null。

七站测点数为：高安屯3、京西5、京阳3、京桥3、京丰1、未来2、上庄2，共19点。编码与原预测输入中的功率部分一致，样例列出了全部字段。

## 3. 输出和缺测

`result_point` 只放浮点数；字符串放 `extra_info`，两者内部均为 `varname / timestamp / value`。同一列表内varname唯一；`event_key` 沿用 `JNH.Fluxcast.Compute`。

| 情况 | dataStatus | 返回数值 |
|---|---|---|
| 19点都有效，合计和偏差可计算，有同刻可用预测 | complete | totalPowerActual、totalPowerDeviation及各站统计 |
| 19点都有效，合计可计算，无同刻可用预测 | complete | totalPowerActual及各站统计；reason为no_matching_forecast |
| 只有1至18点有效 | incomplete | 不返回七站合计和偏差；保留其他完整站出力和计数；missingPoints列出不可用编码 |
| 19点全部不可用 | missing | 保留各站计数和状态；missingPoints列出全部编码 |

**totalPowerActual = 19个功率测点求和；totalPowerDeviation = 同时刻预测值 − 实测总功率，单位均为MW，不是百分比。** 正偏差表示预测偏高。实测无7天预热要求。

完整时还返回各站出力及占比（总功率>0时）、计数、状态、最新完整实测时间；使用新批次计算偏差时返回 `forecastBatchId`。字段不按数组下标区分，统一按varname读取。

部分缺测不补0、不沿用旧值，不把部分合计冒充完整实测。原预测模块的功率补0规则保持原样，两者用途不同。补传同一UTC时刻会更新实测与偏差，不新增重复记录；不会改写预测值。

缺测、单点值不可用、求和或偏差溢出、尚无匹配预测均返回HTTP 200，只省略无法计算的数值，其他结果照常返回，不用null或占位0代替。`extra_info.reason` 为主要原因码，`extra_info.message` 说明具体测点、不可用原因和值摘要及未返回字段；服务日志记录同样的关联信息。合计为0时仍是有效实测，仅占比无法计算；完整性状态不代表所有结果可计算。最新完整实测时间只统计总功率可计算的记录。全部字段和原因码见[实测接口说明](../docs/实测接口说明.md)。

| HTTP | 含义 |
|---|---|
| 400 | 编码表、帧数、时间、JSON或请求头不合法，message说明原因 |
| 404 | 路径不存在 |
| 408 / 413 | 读取请求超时 / 请求过大 |
| 500 | 内部存储或处理错误，应查看日志并重试 |
| 503 | 仅健康检查：存储异常或同步线程退出 |

全部样例位于 [examples](examples/)。这些是**接口说明用的人工数值，不是模型实测成绩**：19点合计1795.0，假设同刻预测1850.0，偏差55.0。在空库中发送完整输入时，应得到 [output_no_forecast.json](examples/output_no_forecast.json)，不能凭该样例得到55.0偏差。

## 4. 如何匹配已有预测

新进程启动后，每5秒只读访问容器内原预测的 `GET /api/v1/fluxcast/compute/latest`，检查96点曲线，归档到独立SQLite数据库。不调用预测POST，也不直接读写原预测缓存。预测服务暂不可用时，仍可返回实测；已有归档可以继续用于匹配。

采用**精确UTC目标时刻**匹配，不找最近时刻、不用请求到达时间、不跨时刻补值。首次实测请求选择“目标时刻前已归档、且覆盖该目标时刻”的最新曲线；选定后固定该批次，后续补传修正实测时仍用它。历史曲线保留，防止latest被新曲线覆盖后迟到实测无据可查。

新预测接口返回 `forecastBatchId` 和 `forecastGeneratedAt`，归档保留这些值，同一ID不得对应不同曲线。生成时间和首次归档时间均须严格早于目标时刻，避免使用事后预测。重新计算即使曲线数值相同也属于新批次，重复GET不创建新批次。旧归档继续使用曲线SHA256及首次归档时间，但不伪造公开ID或生成时间。数据库记录输入、响应及选用批次，便于追溯。

请在预测生成前启动本模块，并在首个预测目标时刻前留出至少一个同步周期及网络处理余量；服务器时钟应同步。健康检查中的 `forecast_sync`、`last_sync_utc_epoch` 用于检查是否持续取得曲线。`no_forecast`/`unavailable` 不妨碍实测求和。轮询无法保证捕获5秒内连续覆盖的每个中间批次，也不能恢复服务停机期间从未读取的旧曲线；此时宁可缺少偏差，不伪造匹配。

## 5. 部署与运行

同容器配置见 [Docker部署运行说明](../docs/Docker部署运行说明.md)。原端口8000不变，新增8002；宿主机配置：

```dotenv
POWER_OBSERVATIONS_BIND=127.0.0.1
POWER_OBSERVATIONS_PORT=8002
POWER_OBSERVATIONS_RUNTIME_DIR=./runtime_observations
```

新目录挂载至 `/app/runtime_observations`，保存 `observations.sqlite3` 及SQLite辅助文件。**与预测runtime分开，不能共用多个实例**。实测库升级保留，启动时自动增加新元数据字段；备份前停止服务，整体备份该目录。功率归零规则首次升级须启用新的预测runtime并补传7天历史，之后同规则容器重建保留缓存。清空实测库会失去旧预测批次、实测记录和固定匹配关系。当前不自动清理归档，运维需监控磁盘并按业务补传期限另行制定保留策略。

一条Compose启动命令运行两个独立进程，其中一个异常退出时容器退出，由重启策略处理。健康检查同时检查原预测latest和新模块health；健康不代表数据完整或已经有可匹配预测。旧镜像不包含本模块，必须重新构建，不能仅替换Compose后直接使用旧镜像。

不使用Docker时可单独启动实测模块（原预测按原方式启动）：

```bash
python -m observations_service.api --host 127.0.0.1 --port 8002
```

实测模块只用Python标准库，不加载torch或模型。运行镜像只复制本目录顶层Python文件，不装入样例、测试或已有数据库。
