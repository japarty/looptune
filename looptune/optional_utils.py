from yeelight import Bulb


# Optional utilities

def yeelight_eow_notification(bulb_ip):
    bulb = Bulb(bulb_ip)
    bulb.turn_on()
    bulb.set_rgb(0, 255, 0)
    bulb.set_brightness(100)