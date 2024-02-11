import unittest

from poc.DelayedPubSub import DelayedPubSub


class TestDelayedPubSub(unittest.TestCase):
    def setUp(self):
        self.delay = 2
        self.pub_sub = DelayedPubSub(self.delay)
        # Sample topics and messages
        self.topic1 = "vehicle_status"
        self.topic2 = "station_updates"
        self.message1 = {"id": 1, "speed": 100}
        self.message2 = {"id": 2, "status": "active"}

    def test_subscription(self):
        """Test subscribing to topics."""
        subscriber_id = "station_1"
        self.pub_sub.subscribe(subscriber_id, self.topic1)
        self.assertIn(subscriber_id, self.pub_sub.subscriptions)
        self.assertIn(self.topic1, self.pub_sub.subscriptions[subscriber_id])

    def test_publish_and_delayed_delivery(self):
        """Test publishing messages and delayed delivery."""
        self.pub_sub.subscribe("station_1", self.topic1)
        # Publish a message to topic1
        self.pub_sub.publish(self.topic1, self.message1)
        # Immediately check for incoming messages (should be empty due to delay)
        self.assertEqual({}, self.pub_sub.get_incoming_messages("station_1"))
        # Advance one step (still no delivery due to delay)
        self.pub_sub.step()
        self.assertEqual({}, self.pub_sub.get_incoming_messages("station_1"))
        # Advance another step (message should now be delivered)
        self.pub_sub.step()
        incoming_messages = self.pub_sub.get_incoming_messages("station_1")
        self.assertIn(self.topic1, incoming_messages)
        self.assertIn(self.message1, incoming_messages[self.topic1])

    def test_multiple_subscribers(self):
        """Test message delivery to multiple subscribers of the same topic."""
        self.pub_sub.subscribe("station_1", self.topic1)
        self.pub_sub.subscribe("station_2", self.topic1)
        # Publish a message to topic1
        self.pub_sub.publish(self.topic1, self.message1)
        # Advance steps to reach the delivery time
        for _ in range(self.delay):
            self.pub_sub.step()
        # Check if both subscribers received the message
        incoming_messages_station_1 = self.pub_sub.get_incoming_messages("station_1")
        incoming_messages_station_2 = self.pub_sub.get_incoming_messages("station_2")
        self.assertIn(self.message1, incoming_messages_station_1[self.topic1])
        self.assertIn(self.message1, incoming_messages_station_2[self.topic1])

    def test_no_subscribers(self):
        """Test the scenario where no subscribers are present for a topic."""
        # Publish a message without any subscribers
        self.pub_sub.publish(self.topic1, self.message1)
        # Advance steps to reach the delivery time
        self.pub_sub.step()
        self.pub_sub.step()
        # Since there are no subscribers, there should be no delivery
        self.assertEqual({}, self.pub_sub.get_incoming_messages("station_1"))
