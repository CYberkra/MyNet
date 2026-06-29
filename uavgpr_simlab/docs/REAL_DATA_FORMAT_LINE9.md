# Line9origin(36).csv 适配格式

已按上传的 `Line9origin(36).csv` 适配解析器。文件头格式：

```text
Number of Samples = 501,,
Time windows (ns) = 700,,
Number of Traces = 2378,,
Trace interval (m) = 0.09093,,
```

之后为逐道存储，每道 501 行，列为：

```text
longitude, latitude, elevation_m, amplitude, flight_height_or_aux
```

GUI 的“实测数据”页可以读取该格式并导出：

- `*_raw.npz`
- `*_raw.png`
- `*_trace_meta.npz`
- `baselines/*_dewow/mean_subtract/gain/svd/fk*`
