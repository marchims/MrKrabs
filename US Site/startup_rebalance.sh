#~/bin/bash

sleep 10
cd "/home/pi/Crypto/Binance_US/"
while :
do
	/home/pi/Crypto/python/bin/python3.6 StartRebalance.py
done
