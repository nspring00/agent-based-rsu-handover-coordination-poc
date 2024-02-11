from collections import deque, defaultdict


class DelayedPubSub:
    def __init__(self, delay):
        self.delay = delay
        self.subscriptions = defaultdict(list)
        self.message_queue = deque([{} for _ in range(delay + 1)], maxlen=delay + 1)

    def subscribe(self, subscriber_id, topic):
        """Subscribe a listener to a topic."""
        if subscriber_id not in self.subscriptions:
            self.subscriptions[subscriber_id] = []
        if topic not in self.subscriptions[subscriber_id]:
            self.subscriptions[subscriber_id].append(topic)

    def publish(self, topic, message):
        """Publish a message to a topic, added to the end of the queue for delivery."""
        # Append the message under its topic in the dictionary at the last position of the deque
        if topic not in self.message_queue[-1]:
            self.message_queue[-1][topic] = []
        self.message_queue[-1][topic].append(message)

    def step(self):
        """Advance one time step, shifting the message queue."""
        # Pop messages from the front of the deque, which are now due for delivery
        self.message_queue.popleft()
        # Append an empty dictionary for the new timestep at the end of the deque
        self.message_queue.append({})

    def get_incoming_messages(self, subscriber_id):
        """Fetch the dictionary of subscribed topics and their messages for the subscriber."""
        incoming_messages = {}
        if subscriber_id in self.subscriptions:
            subscribed_topics = self.subscriptions[subscriber_id]
            for topic in subscribed_topics:
                if topic in self.message_queue[0]:
                    incoming_messages[topic] = self.message_queue[0][topic]
        return incoming_messages
