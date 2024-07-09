from yeelight import Bulb


# Optional utilities

def yeelight_notification(bulb_ip, change_state='on', rgb=False):
    bulb = Bulb(bulb_ip)
    if change_state=='on':
        bulb.turn_on()
        bulb.set_brightness(100)
    elif change_state=='off':
        bulb.turn_off()
    if rgb:
        bulb.set_rgb(*rgb)