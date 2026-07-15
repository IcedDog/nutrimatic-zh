# Nutrimatic 中文版

Nutrimatic 中文版是一个面向中文语料的 Rust 模式搜索器。它支持正则式模式、交集、
乱序搜索，以及拼音、部件、笔画、同字结构和组词补字等中文属性。

## 查询语法

```text
汉字                 字面匹配
.                    任意一个汉字
[春秋]               春或秋
(春|秋)风            分组和选择
?  *  +  {m,n}       重复量词
表达式&表达式        同时匹配同一个完整结果
<天地玄黄>           内部各部分可以乱序
```

中文属性：

```text
@p(shan3)            拼音
@b(氵青)             部件
@h(7-9)              笔画范围
@t(A)@t(A)           相同 ID 匹配同一个字
@z(.士林)            在内置词表中补全单字槽位
```

`.` 是唯一的单汉字通配符。任意多个汉字写作 `.*`，任意一个或多个汉字写作 `.+`。
查询必须匹配完整候选。

## 构建

需要最新版稳定版 Rust：

```console
cargo build --release
```

构建前需在本地准备程序所需的中文属性数据以及对应的解析代码。

## 准备语料

建立索引前，需要先解析语料，生成加权中文记录和审计报告。解析完成后应人工检查报告，
再将记录交给索引命令。英文、数字、表情和标点会作为片段边界，不会连接两侧汉字。

```console
nutrimatic-zh prepare --kind KIND --input SOURCE --output RECORDS.tsv --report AUDIT.json
nutrimatic-zh index --output INDEX.ntri --shard-dir SHARD_DIR RECORDS.tsv...
```

## 查询与网页服务

```console
nutrimatic-zh search --index INDEX.ntri --limit 100 ".*春.*&.{4}"
nutrimatic-zh serve --index INDEX.ntri --bind 127.0.0.1:8080
nutrimatic-zh inspect --index INDEX.ntri --full
```

索引读取器会将索引载入内存，因此需要准备与索引大小相当的可用内存。

## Docker 部署

镜像只包含程序和网页资源，不包含 `.ntri` 索引。运行时将索引所在目录以只读方式挂载到容器的 `/data`。

构建镜像：

```console
docker build -t nutrimatic-zh .
```

使用 Compose 时设置环境文件，然后启动：

```console
docker compose up --build -d
```

Compose 中包含可选的 `cloudflared` 服务。Service URL 设置为：

```text
http://nutrimatic:8080
```

## 许可证

Rust 代码使用 GPL-3.0-or-later。自行准备的中文属性数据和语料仍受各自许可证约束。
