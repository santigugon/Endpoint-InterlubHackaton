
import pandas as pd # type: ignore
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics


def read_file():
    # Cargar los datos
    df = pd.read_csv('orders_preprocessed_top_items.csv')

    # Convertir la columna 'date' a tipo datetime
    df['date'] = pd.to_datetime(df['date'])

    # Agrupar por semana y por 'product_id', sumando solo 'quantity'
    df_weekly = df.groupby([pd.Grouper(key='date', freq='W'), 'product_id'])['quantity'].sum().reset_index()
    df_weekly['week_number'] = (df_weekly['date'] - df_weekly['date'].min()).dt.days // 7
    df_weekly.to_csv("orders_weeks.csv")
    # Mostrar las primeras filas
    return df_weekly

def fitProphet(product_series, confidence):
    product_series['floor'] = 0
    modelo = Prophet(interval_width=confidence)
    modelo.fit(product_series)
    futuro = modelo.make_future_dataframe(periods=5)
    futuro['floor'] = 0
    prediccion = modelo.predict(futuro)
    #Cross-validation
    df_cv = cross_validation(modelo, initial='540 days', period='90 days', horizon='180 days')
    return df_cv

def train_models(orders):
    # List of products to forecast
    productos = ['IVP04039','IVP07165','IVP04009', 'IVP11694']
    #trained_models={}
    predictions={}

    # Loop through each product and plot its forecast
    for i, producto in enumerate(productos):
        # Filter data for the current product
        df_producto = orders[orders["product_id"] == producto].copy()
        df_producto = df_producto.groupby("date")["quantity"].sum().reset_index()
        df_producto.columns = ["ds", "y"]

        # Fit Prophet model
        modelo = Prophet(interval_width=0.80)
        modelo.fit(df_producto)

        # Create future dataframe for predictions
        futuro = modelo.make_future_dataframe(periods=90, freq="W")
        prediccion = modelo.predict(futuro)
        predictions[producto] = prediccion

        #print("Prediccion", prediccion)
    return predictions

def predict_future(date, product, predictions):
    """
    Returns forecast row for the week of the provided date.
    """
    import pandas as pd

    df_pred = predictions.get(product)
    if df_pred is None:
        return {"error": f"No predictions found for product {product}"}

    # Ensure the date is in datetime format
    date = pd.to_datetime(date)

    # Convert 'ds' to datetime just to be safe
    df_pred['ds'] = pd.to_datetime(df_pred['ds'])

    # Get the week number for the input date and for each prediction row
    input_week = date.isocalendar().week
    input_year = date.isocalendar().year

    # Filter for the same ISO week & year
    matching_rows = df_pred[df_pred['ds'].apply(lambda d: d.isocalendar().week == input_week and d.isocalendar().year == input_year)]

    if not matching_rows.empty:
        return matching_rows[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    else:
        return {"message": f"No prediction found for week of {date.date()} (week {input_week}, {input_year})"}

