# import dash
# from dash import html, dcc, clientside_callback
# from dash.dependencies import Input, Output, State
# import pandas as pd
# import sqlalchemy
# import asyncio
# import websockets
# import threading
# import json
# from datetime import datetime, timedelta
# import pytz
# import plotly.graph_objs as go
# import logging

# # ---------------- Logging Setup ----------------
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # ---------------- DB CONFIG ----------------
# DB_CONFIG = {
#     "host": "localhost",
#     "database": "crypto_data",
#     "user": "raviranjan",
#     "password": "",  # Update with secure password in production
#     "port": 5432
# }
# DB_URI = f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
# PAIRS = ["BTCUSDT", "ETHUSDT"]
# INTERVALS = ["1m", "3m", "5m", "15m", "1h", "2h", "4h", "8h", "12h", "1D"]
# LIVE_PRICE = {pair: 0 for pair in PAIRS}
# PREV_LIVE_PRICE = {pair: 0 for pair in PAIRS}  # For green/red coloring
# HISTORICAL_CANDLES_1M = {pair: [] for pair in PAIRS}  # Cached 1m data
# LIVE_CANDLES_1M = {pair: {} for pair in PAIRS}  # 1m live candles
# WEBSOCKET_STATUS = {pair: "Disconnected" for pair in PAIRS}
# LAST_TRADE_TIME = {pair: None for pair in PAIRS}
# IST = pytz.timezone("Asia/Kolkata")

# # TradingView-like styles
# TV_COLORS = {
#     'bg': '#1e222d',
#     'paper': '#1e222d',
#     'grid': '#2a2e39',
#     'up': '#26a69a',
#     'down': '#ef5350',
#     'text': '#ffffff',
#     'border': '#2a2e39'
# }

# # Interval to Pandas resample rule
# INTERVAL_MAP = {
#     "1m": "1min",
#     "3m": "3min",
#     "5m": "5min",
#     "15m": "15min",
#     "1h": "1H",
#     "2h": "2H",
#     "4h": "4H",
#     "8h": "8H",
#     "12h": "12H",
#     "1D": "1D"
# }

# # ---------------- Fetch 1m Historical OHLCV ----------------
# def fetch_ohlcv_1m(pair, limit=1440):  # ~1 day of 1m candles
#     try:
#         engine = sqlalchemy.create_engine(DB_URI)
#         query = f"""
#             SELECT time, open, high, low, close, volume
#             FROM ohlcv_data
#             WHERE pair='{pair}'
#             ORDER BY time DESC
#             LIMIT {limit}
#         """
#         df = pd.read_sql(query, engine)
#         logger.info(f"Fetched {len(df)} 1m rows for {pair} from ohlcv_data")
#         if not df.empty:
#             logger.info(f"Raw timestamp sample for {pair}: {df['time'].iloc[0]}")
#             df['time'] = pd.to_datetime(df['time']).dt.tz_convert(IST).dt.tz_localize(None)
#             logger.info(f"Parsed IST timestamp sample for {pair}: {df['time'].iloc[0]}")
#         return df[::-1]  # Ascending order
#     except Exception as e:
#         logger.error(f"Database error for {pair}: {e}")
#         return pd.DataFrame()
#     finally:
#         engine.dispose()

# # ---------------- Resample to Target Interval ----------------
# def resample_to_interval(df, interval, limit=500):
#     if df.empty:
#         return pd.DataFrame()
#     df.set_index('time', inplace=True)
#     resample_rule = INTERVAL_MAP[interval]
#     df_resampled = df.resample(resample_rule).agg({
#         'open': 'first',
#         'high': 'max',
#         'low': 'min',
#         'close': 'last',
#         'volume': 'sum'
#     }).dropna().reset_index()
#     df_resampled = df_resampled.sort_values('time').tail(limit)
#     logger.info(f"Resampled to {interval}: {len(df_resampled)} candles")
#     return df_resampled

# # ---------------- Preload 1m Historical Data ----------------
# for pair in PAIRS:
#     df = fetch_ohlcv_1m(pair)
#     HISTORICAL_CANDLES_1M[pair] = [
#         {"time": row['time'], "open": float(row['open']), "high": float(row['high']),
#          "low": float(row['low']), "close": float(row['close']), "volume": float(row['volume'])}
#         for _, row in df.iterrows()
#     ]
#     if HISTORICAL_CANDLES_1M[pair]:
#         LIVE_PRICE[pair] = HISTORICAL_CANDLES_1M[pair][-1]['close']
#         PREV_LIVE_PRICE[pair] = LIVE_PRICE[pair]
#         logger.info(f"Initialized {pair} 1m with {len(HISTORICAL_CANDLES_1M[pair])} historical candles")
#     else:
#         logger.warning(f"No 1m historical data for {pair}")

# # ---------------- WebSocket: Aggregate 1m Candles from Trades ----------------
# async def websocket_trades(pair):
#     url = f"wss://stream.binance.com:9443/ws/{pair.lower()}@trade"
#     reconnect_delay = 1
#     while True:
#         try:
#             async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
#                 WEBSOCKET_STATUS[pair] = "Connected"
#                 logger.info(f"WebSocket connected for {pair}")
#                 while True:
#                     msg = json.loads(await ws.recv())
#                     trade_time = datetime.fromtimestamp(msg['T']/1000, IST).replace(tzinfo=None)
#                     price = float(msg['p'])
#                     qty = float(msg['q'])
#                     PREV_LIVE_PRICE[pair] = LIVE_PRICE[pair]
#                     LIVE_PRICE[pair] = price
#                     LAST_TRADE_TIME[pair] = trade_time
#                     logger.debug(f"{pair} trade: price={price:.2f}, time={trade_time}, qty={qty:.2f}")
                    
#                     # Aggregate 1-minute candle
#                     ts = trade_time.replace(second=0, microsecond=0)
#                     if ts not in LIVE_CANDLES_1M[pair]:
#                         LIVE_CANDLES_1M[pair][ts] = {
#                             "time": ts, "open": price, "high": price,
#                             "low": price, "close": price, "volume": qty
#                         }
#                     else:
#                         candle = LIVE_CANDLES_1M[pair][ts]
#                         candle['high'] = max(candle['high'], price)
#                         candle['low'] = min(candle['low'], price)
#                         candle['close'] = price
#                         candle['volume'] += qty
                    
#                     # Clean up old 1m candles
#                     current_time = datetime.now().replace(second=0, microsecond=0)
#                     cutoff = current_time - timedelta(minutes=60)
#                     LIVE_CANDLES_1M[pair] = {ts: c for ts, c in LIVE_CANDLES_1M[pair].items() if ts >= cutoff}
#                     logger.debug(f"Updated {pair} 1m live candles: {len(LIVE_CANDLES_1M[pair])} candles")
#         except Exception as e:
#             WEBSOCKET_STATUS[pair] = f"Disconnected: {str(e)}"
#             logger.error(f"WebSocket error for {pair}: {e}")
#             reconnect_delay = min(reconnect_delay * 2, 60)
#             await asyncio.sleep(reconnect_delay)

# def start_ws_loop():
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     tasks = [websocket_trades(pair) for pair in PAIRS]
#     loop.run_until_complete(asyncio.gather(*tasks))

# threading.Thread(target=start_ws_loop, daemon=True).start()

# # ---------------- Dash App ----------------
# app = dash.Dash(__name__, external_stylesheets=['/assets/custom.css'])
# server = app.server

# # Cache for resampled candles
# CACHED_CANDLES = {pair: {} for pair in PAIRS}  # pair -> interval -> candles
# LAST_FETCH_TIME = {pair: None for pair in PAIRS}

# # TradingView-like layout
# app.layout = html.Div([
#     # Realtime Price Label at Top
#     html.Div([
#         html.Div(id="live-price-label", style={
#             "font-size": "24px", 
#             "font-weight": "bold", 
#             "text-align": "center", 
#             "padding": "10px", 
#             "background": TV_COLORS['border'],
#             "color": TV_COLORS['text'],
#             "height": "50px",
#             "line-height": "50px"
#         })
#     ], style={
#         "position": "fixed", 
#         "top": 0, 
#         "left": 0, 
#         "right": 0, 
#         "z-index": 1001, 
#         "background": TV_COLORS['bg'],
#         "height": "50px"
#     }),

#     # Top Toolbar (below price label)
#     html.Div([
#         html.Div([
#             dcc.Dropdown(
#                 id="pair-dropdown",
#                 options=[{"label": p, "value": p} for p in PAIRS],
#                 value="BTCUSDT",
#                 clearable=False,
#                 style={
#                     "width": "150px", 
#                     "font-family": "Arial, sans-serif",
#                     "min-height": "40px",
#                     "line-height": "40px",
#                     "margin-right": "10px",
#                     "color": "#000000",  # Black text for visibility
#                     "background-color": "#3a3f4a",  # Dark background to match toolbar
#                     "border": "1px solid #ffffff"
#                 }
#             ),
#             dcc.Dropdown(
#                 id="interval-dropdown",
#                 options=[{"label": i, "value": i} for i in INTERVALS],
#                 value="1m",
#                 clearable=False,
#                 style={
#                     "width": "80px", 
#                     "font-family": "Arial, sans-serif",
#                     "min-height": "40px",
#                     "line-height": "40px",
#                     "margin-right": "10px",
#                     "color": "#000000",  # Black text for visibility
#                     "background-color": "#3a3f4a",  # Dark background to match toolbar
#                     "border": "1px solid #ffffff"
#                 }
#             ),
#             html.Span("Candlestick", id="style-dropdown", style={
#                 "margin-left": "10px", 
#                 "color": TV_COLORS['text'],
#                 "line-height": "40px"
#             })
#         ], style={
#             "display": "flex", 
#             "align-items": "center", 
#             "padding": "5px 10px",
#             "height": "50px"
#         }),
#     ], style={
#         "position": "fixed", 
#         "top": "50px", 
#         "left": 0, 
#         "right": 0, 
#         "z-index": 1000, 
#         "background": TV_COLORS['border'],
#         "height": "50px",
#         "box-shadow": "0 2px 5px rgba(0,0,0,0.3)",
#         "position": "relative"  # Anchor dropdown menus
#     }),

#     # Main Chart Area
#     html.Div([
#         dcc.Graph(
#             id="candlestick-chart",
#             style={
#                 "height": "calc(100vh - 170px)", 
#                 "width": "100%", 
#                 "margin-top": "170px", 
#                 "padding": 0, 
#                 "z-index": 999,
#                 "overflow": "hidden"  # Clip chart overflow
#             },
#             config={
#                 'scrollZoom': True,
#                 'modeBarButtonsToAdd': ['zoomin', 'zoomout', 'pan2d', 'autoscale', 'resetaxes', 'drawline', 'drawrect', 'eraseshape'],
#                 'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'hoverClosestCartesian', 'hoverCompareCartesian'],
#                 'doubleClick': 'reset+autosize',
#                 'displaylogo': False,
#                 'displayModeBar': True,
#                 'responsive': True
#             }
#         ),
#         html.Div(id="error-message", style={
#             "color": "red", 
#             "text-align": "center", 
#             "margin-top": "10px", 
#             "font-size": "18px"
#         })
#     ], style={
#         "flex": 1, 
#         "background": TV_COLORS['bg'], 
#         "padding": 0, 
#         "margin": 0,
#         "min-height": "100vh"
#     }),

#     dcc.Interval(id="update-interval", interval=500, n_intervals=0)
# ], style={
#     "background": TV_COLORS['bg'], 
#     "color": TV_COLORS['text'], 
#     "font-family": "Arial, sans-serif", 
#     "height": "100vh", 
#     "margin": 0
# })

# # Client-side callback to log zoom/pan events
# clientside_callback(
#     """
#     function(relayoutData) {
#         if (relayoutData && (relayoutData['xaxis.range[0]'] || relayoutData['yaxis.range[0]'])) {
#             console.log('Zoom/Pan event:', relayoutData);
#         }
#         return window.dash_clientside.no_update;
#     }
#     """,
#     Output("candlestick-chart", "id"),
#     Input("candlestick-chart", "relayoutData")
# )

# # Client-side callback to style dropdown menus with delay
# clientside_callback(
#     """
#     function(n) {
#         setTimeout(function() {
#             var dropdownMenus = document.querySelectorAll('.Select-menu-outer, .Select-menu');
#             dropdownMenus.forEach(function(menu) {
#                 menu.style.zIndex = '1002 !important';
#                 menu.style.maxHeight = '100px !important';
#                 menu.style.overflowY = 'auto';
#                 menu.style.position = 'absolute';
#                 menu.style.top = '100% !important';
#                 menu.style.left = '0';
#                 menu.style.backgroundColor = '#3a3f4a';
#                 menu.style.color = '#000000';
#                 menu.style.border = '1px solid #ffffff';
#                 menu.style.boxShadow = '0 2px 5px rgba(0,0,0,0.5)';
#             });
#         }, 100);  // 100ms delay to ensure menu is rendered
#         return window.dash_clientside.no_update;
#     }
#     """,
#     Output("candlestick-chart", "id"),
#     Input("update-interval", "n_intervals")
# )

# # ---------------- Update Callback ----------------
# @app.callback(
#     [Output("candlestick-chart", "figure"),
#      Output("live-price-label", "children"),
#      Output("live-price-label", "style"),
#      Output("error-message", "children")],
#     [Input("pair-dropdown", "value"),
#      Input("interval-dropdown", "value"),
#      Input("update-interval", "n_intervals")],
#     [State("candlestick-chart", "figure")]
# )
# def update_chart(pair, interval, n, current_figure):
#     try:
#         # Check if we need to fetch new data
#         current_time = datetime.now()
#         if LAST_FETCH_TIME[pair] is None or (current_time - LAST_FETCH_TIME[pair]).total_seconds() > 300:  # Refresh every 5min
#             df_1m = fetch_ohlcv_1m(pair, limit=1440)
#             if not df_1m.empty:
#                 HISTORICAL_CANDLES_1M[pair] = [
#                     {"time": row['time'], "open": float(row['open']), "high": float(row['high']),
#                      "low": float(row['low']), "close": float(row['close']), "volume": float(row['volume'])}
#                     for _, row in df_1m.iterrows()
#                 ]
#                 LAST_FETCH_TIME[pair] = current_time
#                 CACHED_CANDLES[pair] = {}  # Clear cache on new fetch

#         # Use cached resampled data if available
#         if interval not in CACHED_CANDLES[pair]:
#             df_1m = pd.DataFrame(HISTORICAL_CANDLES_1M[pair])
#             df_resampled = resample_to_interval(df_1m, interval, limit=500)
#             CACHED_CANDLES[pair][interval] = [
#                 {"time": row['time'], "open": float(row['open']), "high": float(row['high']),
#                  "low": float(row['low']), "close": float(row['close']), "volume": float(row['volume'])}
#                 for _, row in df_resampled.iterrows()
#             ]

#         # Merge with live 1m candles (resample live to target interval)
#         live_1m_candles = sorted(LIVE_CANDLES_1M[pair].values(), key=lambda x: x['time'])
#         if live_1m_candles:
#             df_live_1m = pd.DataFrame(live_1m_candles)
#             df_live_1m['time'] = pd.to_datetime(df_live_1m['time'])
#             df_live_1m.set_index('time', inplace=True)
#             df_live_resampled = df_live_1m.resample(INTERVAL_MAP[interval]).agg({
#                 'open': 'first',
#                 'high': 'max',
#                 'low': 'min',
#                 'close': 'last',
#                 'volume': 'sum'
#             }).dropna().reset_index()
#             live_candles = [
#                 {"time": row['time'], "open": float(row['open']), "high": float(row['high']),
#                  "low": float(row['low']), "close": float(row['close']), "volume": float(row['volume'])}
#                 for _, row in df_live_resampled.iterrows()
#             ]
#         else:
#             live_candles = []

#         # Filter live candles after last historical
#         last_historical_time = CACHED_CANDLES[pair][interval][-1]['time'] if CACHED_CANDLES[pair].get(interval) else None
#         if last_historical_time:
#             live_candles = [c for c in live_candles if c['time'] > last_historical_time]
#         candles = CACHED_CANDLES[pair][interval] + live_candles
#         logger.info(f"Rendering {pair} {interval} with {len(candles)} candles (historical: {len(CACHED_CANDLES[pair][interval])}, live: {len(live_candles)})")

#         if not candles:
#             logger.warning(f"No candles available for {pair} {interval}")
#             fig = go.Figure()
#             fig.update_layout(
#                 title=f"{pair} - {interval} Chart",
#                 xaxis_rangeslider_visible=False,
#                 plot_bgcolor=TV_COLORS['bg'],
#                 paper_bgcolor=TV_COLORS['paper'],
#                 font_color=TV_COLORS['text'],
#                 xaxis=dict(title="Time (IST)", type='date', showgrid=True, gridcolor=TV_COLORS['grid']),
#                 yaxis=dict(title="Price (USDT)", showgrid=True, gridcolor=TV_COLORS['grid']),
#                 annotations=[dict(
#                     text="No data available. Check database or WebSocket connection.",
#                     xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
#                     font=dict(size=20, color="red")
#                 )]
#             )
#             return fig, "Live Price: N/A", {"font-size": "24px", "font-weight": "bold", "text-align": "center", "padding": "10px", "background": TV_COLORS['border'], "color": TV_COLORS['text'], "height": "50px", "line-height": "50px"}, f"Error: No data available. WebSocket: {WEBSOCKET_STATUS[pair]}"

#         # Prepare data
#         times = [c['time'] for c in candles]
#         opens = [c['open'] for c in candles]
#         highs = [c['high'] for c in candles]
#         lows = [c['low'] for c in candles]
#         closes = [c['close'] for c in candles]

#         # Hover text
#         hover_texts = [
#             f"Time: {t.strftime('%Y-%m-%d %H:%M:%S IST')}<br>Open: {o:.2f}<br>High: {h:.2f}<br>Low: {l:.2f}<br>Close: {c:.2f}"
#             for t, o, h, l, c in zip(times, opens, highs, lows, closes)
#         ]

#         # Create figure
#         fig = go.Figure()
#         fig.add_trace(go.Candlestick(
#             x=times, open=opens, high=highs, low=lows, close=closes,
#             increasing_line_color=TV_COLORS['up'], decreasing_line_color=TV_COLORS['down'],
#             increasing_fillcolor=TV_COLORS['up'], decreasing_fillcolor=TV_COLORS['down'],
#             name="Price", showlegend=False,
#             hovertext=hover_texts, hoverinfo="text"
#         ))

#         # Price indicator line (horizontal line at live price)
#         last_price = LIVE_PRICE.get(pair, closes[-1] if closes else 0)
#         fig.add_hline(y=last_price, line_dash="dot", line_color="yellow", opacity=0.8,
#                       annotation_text=f"Live Price: {last_price:.2f}", annotation_position="right top")

#         # TradingView-like layout
#         fig.update_layout(
#             title=f"{pair} - {interval} Chart",
#             xaxis_rangeslider_visible=False,
#             plot_bgcolor=TV_COLORS['bg'],
#             paper_bgcolor=TV_COLORS['paper'],
#             font_color=TV_COLORS['text'],
#             yaxis_side="right",
#             hovermode="x unified",
#             dragmode='zoom',
#             xaxis=dict(
#                 fixedrange=False,
#                 gridcolor=TV_COLORS['grid'],
#                 type='date',
#                 tickmode='auto',
#                 nticks=10,
#                 title="Time (IST)",
#                 tickformatstops=[
#                     dict(dtickrange=[None, 60000], value="%H:%M:%S IST"),
#                     dict(dtickrange=[60000, 3600000], value="%H:%M IST"),
#                     dict(dtickrange=[3600000, 86400000], value="%Y-%m-%d %H:%M"),
#                     dict(dtickrange=[86400000, None], value="%Y-%m-%d")
#                 ]
#             ),
#             yaxis=dict(
#                 fixedrange=False,
#                 gridcolor=TV_COLORS['grid'],
#                 title="Price (USDT)",
#                 autorange=True
#             ),
#             showlegend=False,
#             margin=dict(l=0, r=0, t=30, b=0)
#         )
#         fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=TV_COLORS['grid'])
#         fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=TV_COLORS['grid'], title="Price (USDT)")

#         # WebSocket status
#         error_msg = ""
#         last_trade_time = LAST_TRADE_TIME.get(pair)
#         if last_trade_time:
#             time_since_last_trade = (datetime.now() - last_trade_time).total_seconds()
#             if time_since_last_trade > 10:
#                 error_msg = f"WebSocket stale: last trade {time_since_last_trade:.1f}s ago"
#         if WEBSOCKET_STATUS[pair] != "Connected":
#             error_msg = f"WebSocket: {WEBSOCKET_STATUS[pair]}"

#         # Realtime price label at top with green/red color
#         live_price_text = f"Live Price: {last_price:.2f} USDT"
#         price_color = TV_COLORS['up'] if last_price >= PREV_LIVE_PRICE[pair] else TV_COLORS['down']
#         live_price_style = {
#             "font-size": "24px",
#             "font-weight": "bold",
#             "text-align": "center",
#             "padding": "10px",
#             "background": TV_COLORS['border'],
#             "color": price_color,
#             "height": "50px",
#             "line-height": "50px"
#         }

#         return fig, live_price_text, live_price_style, error_msg

#     except Exception as e:
#         logger.error(f"Error rendering chart for {pair} {interval}: {e}")
#         fig = go.Figure()
#         fig.update_layout(
#             title=f"{pair} - {interval} Chart",
#             xaxis_rangeslider_visible=False,
#             plot_bgcolor=TV_COLORS['bg'],
#             paper_bgcolor=TV_COLORS['paper'],
#             font_color=TV_COLORS['text'],
#             xaxis=dict(title="Time (IST)", type='date', showgrid=True, gridcolor=TV_COLORS['grid']),
#             yaxis=dict(title="Price (USDT)", showgrid=True, gridcolor=TV_COLORS['grid']),
#             annotations=[dict(
#                 text=f"Error rendering chart: {str(e)}",
#                 xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
#                 font=dict(size=20, color="red")
#             )]
#         )
#         return fig, "Live Price: N/A", {"font-size": "24px", "font-weight": "bold", "text-align": "center", "padding": "10px", "background": TV_COLORS['border'], "color": TV_COLORS['text'], "height": "50px", "line-height": "50px"}, f"Error: {str(e)}. WebSocket: {WEBSOCKET_STATUS[pair]}"

# # ---------------- Run ----------------
# if __name__ == "__main__":
#     print("🚀 TradingView-Style Dashboard with Multi-Timeframe Support running on http://127.0.0.1:8050")
#     app.run(debug=False, port=8050)



#!/usr/bin/env python3
"""
dashboard_api.py - FastAPI Backend for Trading Dashboard
Serves real-time data from TimescaleDB with IST timezone
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from scripts.database import Database
from datetime import datetime, timedelta, timezone
import pytz
import os
from dotenv import load_dotenv
from typing import Optional, List
import uvicorn
from fastapi import WebSocket, WebSocketDisconnect
from typing import List
import asyncio

import logging
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="Crypto Trading Dashboard API",
    description="Real-time cryptocurrency data API with IST timezone",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
db = Database()

# IST Timezone
IST = pytz.timezone('Asia/Kolkata')

def convert_to_ist(dt):
    """Convert datetime to IST"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST).isoformat()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Crypto Trading Dashboard API",
        "version": "1.0.0",
        "timezone": "IST (Asia/Kolkata)",
        "docs": "/docs"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    db_status = db.test_connection()
    return {
        "status": "healthy" if db_status else "unhealthy",
        "database": "connected" if db_status else "disconnected",
        "timestamp": convert_to_ist(datetime.now(timezone.utc)),
        "timezone": "IST"
    }

@app.get("/api/candles")
async def get_candles(
    pair: str = Query("BTCUSDT", description="Trading pair"),
    timeframe: str = Query("1h", description="Timeframe (1m, 5m, 15m, 1h, 4h, 1D)"),
    limit: int = Query(500, ge=1, le=5000, description="Number of candles"),
    start_time: Optional[str] = Query(None, description="Start time (ISO format)"),
    end_time: Optional[str] = Query(None, description="End time (ISO format)")
):
    """
    Get OHLCV candles - aggregates from 1m data for other timeframes
    """
    conn = None
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Map timeframe to interval
        interval_map = {
            '1m': '1 minute',
            '3m': '3 minutes',
            '5m': '5 minutes',
            '15m': '15 minutes',
            '30m': '30 minutes',
            '1h': '1 hour',
            '2h': '2 hours',
            '4h': '4 hours',
            '6h': '6 hours',
            '8h': '8 hours',
            '12h': '12 hours',
            '1d': '1 day',
            '3d': '3 days',
            '1w': '1 week'
        }
        
        interval = interval_map.get(timeframe, '1 hour')
        
        # If requesting 1m data, query directly
        if timeframe == '1m':
            query = """
                SELECT 
                    time, open, high, low, close, volume,
                    quote_volume, num_trades, taker_buy_base, taker_buy_quote
                FROM ohlcv_data
                WHERE pair = %s AND timeframe = '1m'
            """
            params = [pair]
            
            if start_time:
                query += " AND time >= %s"
                params.append(start_time)
            if end_time:
                query += " AND time <= %s"
                params.append(end_time)
            
            query += " ORDER BY time ASC LIMIT %s"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
        else:
            # Aggregate from 1m data using time_bucket
            query = f"""
                SELECT 
                    time_bucket('{interval}', time) as bucket_time,
                    (array_agg(open ORDER BY time))[1] as open,
                    MAX(high) as high,
                    MIN(low) as low,
                    (array_agg(close ORDER BY time DESC))[1] as close,
                    SUM(volume) as volume,
                    SUM(quote_volume) as quote_volume,
                    SUM(num_trades) as num_trades,
                    SUM(taker_buy_base) as taker_buy_base,
                    SUM(taker_buy_quote) as taker_buy_quote
                FROM ohlcv_data
                WHERE pair = %s AND timeframe = '1m'
            """
            params = [pair]
            
            if start_time:
                query += " AND time >= %s"
                params.append(start_time)
            if end_time:
                query += " AND time <= %s"
                params.append(end_time)
            
            query += f" GROUP BY bucket_time ORDER BY bucket_time ASC LIMIT %s"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
        
        # Format candles
        candles = []
        for row in rows:
            candles.append({
                'time': convert_to_ist(row[0]),
                'timestamp': int(row[0].timestamp() * 1000) if row[0] else 0,
                'open': float(row[1]) if row[1] else 0,
                'high': float(row[2]) if row[2] else 0,
                'low': float(row[3]) if row[3] else 0,
                'close': float(row[4]) if row[4] else 0,
                'volume': float(row[5]) if row[5] else 0,
                'quote_volume': float(row[6]) if row[6] else 0,
                'num_trades': int(row[7]) if row[7] else 0,
                'taker_buy_base': float(row[8]) if row[8] else 0,
                'taker_buy_quote': float(row[9]) if row[9] else 0
            })
        
        cursor.close()
        
        return {
            'pair': pair,
            'timeframe': timeframe,
            'timezone': 'IST',
            'candles': candles,
            'count': len(candles)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            db.return_connection(conn)
            
@app.get("/api/stats")
async def get_stats(pair: str = Query("BTCUSDT", description="Trading pair")):
    """Get 24h statistics in IST timezone"""
    conn = None
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Get current time in UTC
        now_utc = datetime.now(timezone.utc)
        day_ago = now_utc - timedelta(hours=24)
        
        # Latest price and 24h data
        query = """
            WITH latest AS (
                SELECT close, time
                FROM ohlcv_data
                WHERE pair = %s AND timeframe = '1m'
                ORDER BY time DESC
                LIMIT 1
            ),
            day_ago_price AS (
                SELECT close
                FROM ohlcv_data
                WHERE pair = %s AND timeframe = '1m'
                  AND time <= %s
                ORDER BY time DESC
                LIMIT 1
            ),
            stats_24h AS (
                SELECT 
                    MAX(high) as high_24h,
                    MIN(low) as low_24h,
                    SUM(volume) as volume_24h,
                    SUM(quote_volume) as quote_volume_24h,
                    COUNT(*) as trades_24h
                FROM ohlcv_data
                WHERE pair = %s 
                  AND timeframe = '1m'
                  AND time >= %s
            )
            SELECT 
                latest.close as latest_price,
                latest.time as latest_time,
                day_ago_price.close as price_24h_ago,
                stats_24h.*
            FROM latest, day_ago_price, stats_24h
        """
        
        cursor.execute(query, (pair, pair, day_ago, pair, day_ago))
        row = cursor.fetchone()
        
        if row and row[0] and row[2]:
            latest_price = float(row[0])
            price_24h_ago = float(row[2])
            change_24h = ((latest_price - price_24h_ago) / price_24h_ago) * 100
            change_24h_value = latest_price - price_24h_ago
        else:
            latest_price = 0
            change_24h = 0
            change_24h_value = 0
        
        stats = {
            'pair': pair,
            'latest_price': latest_price,
            'latest_time': convert_to_ist(row[1]) if row[1] else None,
            'change_24h_percent': round(change_24h, 2),
            'change_24h_value': round(change_24h_value, 2),
            'high_24h': float(row[3]) if row[3] else 0,
            'low_24h': float(row[4]) if row[4] else 0,
            'volume_24h': float(row[5]) if row[5] else 0,
            'quote_volume_24h': float(row[6]) if row[6] else 0,
            'trades_24h': int(row[7]) if row[7] else 0,
            'timezone': 'IST'
        }
        
        cursor.close()
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            db.return_connection(conn)

@app.get("/api/indicators")
async def get_indicators(
    pair: str = Query("BTCUSDT", description="Trading pair"),
    timeframe: str = Query("1h", description="Timeframe"),
    limit: int = Query(200, ge=50, le=1000, description="Number of candles")
):
    """Calculate technical indicators (MA, EMA, RSI, MACD, Bollinger Bands)"""
    conn = None
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # FIXED: Simpler query without window functions that might fail
        query = """
            SELECT 
                time,
                close,
                high,
                low,
                volume
            FROM ohlcv_data
            WHERE pair = %s AND timeframe = %s
            ORDER BY time DESC
            LIMIT %s
        """
        
        cursor.execute(query, (pair, timeframe, limit * 2))  # Get more for MA calculation
        rows = cursor.fetchall()
        
        if not rows:
            cursor.close()
            return {
                'pair': pair,
                'timeframe': timeframe,
                'timezone': 'IST',
                'indicators': []
            }
        
        # Calculate indicators in Python
        indicators = []
        closes = [float(row[1]) for row in reversed(rows)]
        
        for idx, row in enumerate(reversed(rows[-limit:])):  # Only return 'limit' candles
            ma7 = sum(closes[max(0, idx-6):idx+1]) / min(7, idx+1) if idx >= 0 else closes[idx]
            ma20 = sum(closes[max(0, idx-19):idx+1]) / min(20, idx+1) if idx >= 19 else None
            ma50 = sum(closes[max(0, idx-49):idx+1]) / min(50, idx+1) if idx >= 49 else None
            
            # Bollinger Bands
            if idx >= 19:
                ma20_val = ma20
                stddev = (sum((c - ma20_val)**2 for c in closes[max(0, idx-19):idx+1]) / 20) ** 0.5
                bb_upper = ma20_val + (2 * stddev)
                bb_lower = ma20_val - (2 * stddev)
            else:
                bb_upper = None
                bb_lower = None
            
            indicators.append({
                'time': convert_to_ist(row[0]),
                'timestamp': int(row[0].timestamp() * 1000) if row[0] else 0,
                'close': float(row[1]) if row[1] else 0,
                'high': float(row[2]) if row[2] else 0,
                'low': float(row[3]) if row[3] else 0,
                'volume': float(row[4]) if row[4] else 0,
                'ma7': round(ma7, 2),
                'ma20': round(ma20, 2) if ma20 else None,
                'ma50': round(ma50, 2) if ma50 else None,
                'ma100': None,  # Would need more data
                'ma200': None,  # Would need more data
                'bb_upper': round(bb_upper, 2) if bb_upper else None,
                'bb_middle': round(ma20, 2) if ma20 else None,
                'bb_lower': round(bb_lower, 2) if bb_lower else None
            })
        
        cursor.close()
        
        return {
            'pair': pair,
            'timeframe': timeframe,
            'timezone': 'IST',
            'indicators': indicators
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            db.return_connection(conn)

@app.get("/api/orderbook")
async def get_orderbook_summary(pair: str = Query("BTCUSDT")):
    """Get order book summary from recent trades"""
    conn = None
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Get buy/sell pressure from taker data
        query = """
            SELECT 
                AVG(taker_buy_base) as avg_buy_volume,
                AVG(volume - taker_buy_base) as avg_sell_volume,
                SUM(taker_buy_base) as total_buy,
                SUM(volume - taker_buy_base) as total_sell
            FROM ohlcv_data
            WHERE pair = %s 
              AND timeframe = '1m'
              AND time >= NOW() - INTERVAL '1 hour'
        """
        
        cursor.execute(query, (pair,))
        row = cursor.fetchone()
        
        total_buy = float(row[2]) if row[2] else 0
        total_sell = float(row[3]) if row[3] else 0
        total = total_buy + total_sell
        
        buy_percentage = (total_buy / total * 100) if total > 0 else 50
        sell_percentage = (total_sell / total * 100) if total > 0 else 50
        
        cursor.close()
        
        return {
            'pair': pair,
            'buy_volume': total_buy,
            'sell_volume': total_sell,
            'buy_percentage': round(buy_percentage, 2),
            'sell_percentage': round(sell_percentage, 2),
            'pressure': 'bullish' if buy_percentage > 55 else 'bearish' if buy_percentage < 45 else 'neutral',
            'timezone': 'IST',
            'updated_at': convert_to_ist(datetime.now(timezone.utc))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            db.return_connection(conn)

@app.get("/api/database/stats")
async def get_database_stats():
    """Get overall database statistics"""
    try:
        stats = db.get_statistics()
        
        if not stats:
            raise HTTPException(status_code=500, detail="Could not retrieve statistics")
        
        formatted_stats = {
            'total_rows': stats['total_rows'],
            'db_size': stats['db_size'],
            'timezone': 'IST',
            'pairs': []
        }
        
        for row in stats['stats']:
            formatted_stats['pairs'].append({
                'pair': row[0],
                'timeframe': row[1],
                'count': row[2],
                'first_candle': convert_to_ist(row[3]) if row[3] else None,
                'last_candle': convert_to_ist(row[4]) if row[4] else None
            })
        
        return formatted_stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/timeframes")
async def get_available_timeframes():
    """Get list of available timeframes"""
    return {
        'timeframes': [
            {'value': '1m', 'label': '1 Minute', 'category': 'short'},
            {'value': '3m', 'label': '3 Minutes', 'category': 'short'},
            {'value': '5m', 'label': '5 Minutes', 'category': 'short'},
            {'value': '15m', 'label': '15 Minutes', 'category': 'short'},
            {'value': '30m', 'label': '30 Minutes', 'category': 'medium'},
            {'value': '1h', 'label': '1 Hour', 'category': 'medium'},
            {'value': '2h', 'label': '2 Hours', 'category': 'medium'},
            {'value': '4h', 'label': '4 Hours', 'category': 'medium'},
            {'value': '6h', 'label': '6 Hours', 'category': 'long'},
            {'value': '8h', 'label': '8 Hours', 'category': 'long'},
            {'value': '12h', 'label': '12 Hours', 'category': 'long'},
            {'value': '1d', 'label': '1 Day', 'category': 'long'},
            {'value': '3d', 'label': '3 Days', 'category': 'long'},
            {'value': '1w', 'label': '1 Week', 'category': 'long'}
        ]
    }

@app.get("/api/pairs")
async def get_available_pairs():
    """Get list of available trading pairs"""
    conn = None
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT DISTINCT pair
            FROM ohlcv_data
            ORDER BY pair
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        pairs = [row[0] for row in rows]
        
        cursor.close()
        
        return {
            'pairs': pairs,
            'count': len(pairs)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            db.return_connection(conn)
            
# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time price updates
    Sends latest candle data every 5 seconds
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Fetch latest candles for all pairs
            conn = db.get_connection()
            cursor = conn.cursor()
            
            query = """
                SELECT DISTINCT ON (pair)
                    pair,
                    time,
                    close,
                    high,
                    low,
                    volume
                FROM ohlcv_data
                WHERE timeframe = '1m'
                ORDER BY pair, time DESC
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            data = {
                'type': 'price_update',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'prices': []
            }
            
            for row in rows:
                data['prices'].append({
                    'pair': row[0],
                    'time': convert_to_ist(row[1]),
                    'price': float(row[2]) if row[2] else 0,
                    'high': float(row[3]) if row[3] else 0,
                    'low': float(row[4]) if row[4] else 0,
                    'volume': float(row[5]) if row[5] else 0
                })
            
            cursor.close()
            db.return_connection(conn)
            
            # Send to all connected clients
            await manager.broadcast(data)
            
            # Wait 5 seconds
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    print("="*70)
    print("🚀 Crypto Trading Dashboard API (FastAPI)")
    print("="*70)
    print(f"🌍 API Server: http://localhost:8000")
    print(f"📚 API Docs: http://localhost:8000/docs")
    print(f"🔧 ReDoc: http://localhost:8000/redoc")
    print(f"🕐 Timezone: IST (Asia/Kolkata)")
    print("="*70)
    
    # Test database
    if db.test_connection():
        print("✅ Database connected")
    else:
        print("❌ Database connection failed")
    
    print("\n🎯 Available Endpoints:")
    print("  • GET  /api/health - Health check")
    print("  • GET  /api/candles - OHLCV candles (IST)")
    print("  • GET  /api/stats - 24h statistics")
    print("  • GET  /api/indicators - Technical indicators")
    print("  • GET  /api/orderbook - Order book summary")
    print("  • GET  /api/pairs - Available pairs")
    print("  • GET  /api/timeframes - Available timeframes")
    print("  • GET  /api/database/stats - Database info")
    print("="*70 + "\n")
    
    uvicorn.run(
        "dashboard:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )