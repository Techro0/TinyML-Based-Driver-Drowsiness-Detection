import paho.mqtt.client as mqtt

BROKER="test.mosquitto.org"
TOPIC="tinyml/drowsy/alert"

def on_message(client, userdata, msg):
    print("Message:", msg.topic, msg.payload.decode())

c = mqtt.Client()
c.connect(BROKER, 1883, 60)
c.subscribe(TOPIC)
c.on_message = on_message
print("Listening on", TOPIC)
c.loop_forever()