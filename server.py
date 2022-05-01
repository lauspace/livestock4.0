import paho.mqtt.client as mqtt 
import datetime

xtime=[]
ydata=[]

def on_message(mqttc, obj, msg,):
    # print(msg.topic + " " + str(msg.payload))
    payload = str(msg.payload)
    payload_split=payload.split("'")
    print(msg.topic + " Payload -> " + payload_split[1])

    ydata.append(payload_split[1])
    now=datetime.datetime.now()
    xtime.append(str(now.hour) + ":" + str(now.minute) + ":" + str(now.second))

    print(xtime)
    print(ydata)
    
try:
    mqttc = mqtt.Client()
    mqttc.on_message = on_message

    mqttc.connect("192.168.1.108", 1883, 60)
    mqttc.subscribe("sensor1", 0)

    mqttc.loop_forever()

except KeyboardInterrupt:
    print ("Received topics:")
