# QMChart
QMChart is Quantitative Momentum Chart visualization tool

# How to use
  clone and run 
  
    python ./QuantitativeMomentx/qmchart.py

# Functions
This script displays an interactive price chart for a specified stock using OpenCV.

It supports zooming, panning, and toggling moving averages (MA).

It also integrates with a background thread to update stock data.

Clicking the "QM" label starts a background process to compute High Quality Momentum (HQM) scores.

The stock code can be edited via a dialog box triggered by clicking the title.
     hover tooltip, showing date, price, and indicator values.
     Plot Options menu  
     QM label
     moving averages (MA) lines.
     X ticks, Y ticks
     The price subplot
     MACD subplot, DIF, DEA.
     KDJ subplot, K, D, J.
     golden cross (green), dead cross (red)
     schema1, show green triangle for Stock buy points, based on MACD, DIF crossing DEA, and only when price is above MA114, 
             Show red triangle for Stock sell points when price is below MA114. Check https://www.youtube.com/watch?v=K2u31j4R7-s&t=979s. 
     schema2, QM subplot shows QM data  
     schema3, show buy/sell points based on OBV volume indicator, check https://www.youtube.com/watch?v=4YXQRdLFYNc
              draw red flags for bearish and green for bullish divergences
     schema4, check the stock bottom with RSI Oversold Bounce, show blue triangle if it's bottom
             Draw the stock top with RSI Overbought Reversal, show orange triangle if it's top