
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
    productos = [    'IVP04039',
    'IVP07165',
    'IVP04009',
    'IVP11694',
    'IVP11159',
    'IVP11162',
    'IVP07331',
    'IVP11479',
    'IVP07169']
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


import pandas as pd

def give_actual_demand(date, product_id):
    # Convert input date to datetime
    date = pd.to_datetime(date)

    # Check if the date is for a year beyond 2024
    if date.year > 2024:
        return {"error": "Year must be 2024 or earlier"}
    
    # Load the necessary datasets
    df_weekly = pd.read_csv('orders_weeks.csv')
    df_preprocessed = pd.read_csv('orders_preprocessed_top_items.csv')
    
    # Convert 'date' columns to datetime type for both dataframes
    df_weekly['date'] = pd.to_datetime(df_weekly['date'])
    df_preprocessed['date'] = pd.to_datetime(df_preprocessed['date'])
    
    # Get the week and year of the input date
    input_week = date.isocalendar().week
    input_year = date.isocalendar().year

    # Get the 'unit' value for the product from orders_preprocessed_top_items
    unit_data = df_preprocessed[(df_preprocessed['product_id'] == product_id) & 
                                (df_preprocessed['date'].apply(lambda d: d.isocalendar().week == input_week and d.isocalendar().year == input_year))]
    
    if not unit_data.empty:
        unit = unit_data.iloc[0]['unit']  # Get the first matching unit
    else:
        unit = None  # If no unit found, set as None
    
    # Filter df_weekly for matching week, year, and product_id
    matching_rows = df_weekly[
        (df_weekly['date'].apply(lambda d: d.isocalendar().week == input_week and d.isocalendar().year == input_year)) & 
        (df_weekly['product_id'] == product_id)
    ]
    
    # If no matching rows in df_weekly, return a message
    if matching_rows.empty:
        return {"message": f"No estimate found for week of {date.date()} (week {input_week}, {input_year})"}
    
    # Now return the data, including the 'unit' retrieved from the other dataset
    matching_rows['unit'] = unit  # Add the unit value to the result
    return matching_rows[['date', 'quantity', 'unit']]
