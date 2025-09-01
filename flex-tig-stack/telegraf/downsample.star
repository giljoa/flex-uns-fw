def apply(metric):
    if "data" in metric.fields:
        metric.fields["data"] = metric.fields["data"][::50]  # Select every 50th row
    return metric