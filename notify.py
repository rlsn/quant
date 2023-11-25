from win10toast import ToastNotifier
from futu import *
import time
from datetime import datetime
import signal

def get_price(quote_ctx, code):
    ret, dat = quote_ctx.get_market_snapshot([code])
    if ret == RET_OK:
        return dat['last_price'][0]
    else:
        print('error:', dat)
    return 

def main():
    stock_code = "HK.09988"
    cond = lambda price: price<82.5

    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    def keyboardInterruptHandler(signal, frame):
        quote_ctx.close()
        exit(0)

    signal.signal(signal.SIGINT, keyboardInterruptHandler)
    toast = ToastNotifier()
    while(1):
        price = get_price(quote_ctx,stock_code)
        if cond(price):
            toast.show_toast(
                "Notification",
                f"{stock_code} price: {price}",
                duration = 2,
                threaded = True,
            )
        time.sleep(5)
        print(f"tick {datetime.now().strftime('%H:%M:%S')}| {stock_code} price: {price}")

if __name__=="__main__":
    main()
