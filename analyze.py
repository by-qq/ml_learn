import io


import pandas as pd
from fastapi import FastAPI,UploadFile
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt


app = FastAPI()

@app.post("/salary_by_district")
async def salary_by_district(file: UploadFile):
    file_bytes = await file.read()
    data = pd.read_csv(io.BytesIO(file_bytes))
    data_mean = data.groupby("district").agg({"salary": "mean"})
    data_max = data.groupby("district").agg({"salary": "max"})
    data_min = data.groupby("district").agg({"salary": "min"})
    data_total = pd.concat([data_mean, data_max, data_min], axis=1)
    data_total.columns = ['salary_mean', 'salary_max', 'salary_min']
    # 直接创建 BytesIO 对象并写入 Excel
    output = io.BytesIO()
    data_total.to_excel(output, index=True)
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=salary_report.xlsx"}
    )
@app.post("/company_count_chart")
async def company_count_chart(file: UploadFile):
    # 设置字体为微软雅黑
    import matplotlib
    matplotlib.rc("font", family='Microsoft YaHei')

    file_bytes = await file.read()
    data = pd.read_csv(io.BytesIO(file_bytes))
    company_count = data.groupby("district").agg({"companySize": "count"})
    company_count.plot(kind="bar")

    output = io.BytesIO()
    plt.savefig(output,format="png")
    output.seek(0)
    plt.close()

    return StreamingResponse(
        output,
        media_type="image/png"
    )


if __name__ == "__main__":


    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
