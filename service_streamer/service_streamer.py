# coding=utf-8
# Created by Meteorix at 2019/7/13
from typing import List
from redis import Redis
from queue import Queue, Empty
import os
import uuid
import json
import time
import threading
import weakref


class Future(object):
    def __init__(self, task_id, task_size, future_cache_ref):
        self._id = task_id
        self._size = task_size
        self._future_cache_ref = future_cache_ref
        self._outputs = []
        self._finish_event = threading.Event()

    def result(self, timeout=None):
        finished = self._finish_event.wait(timeout)

        if not finished:
            raise TimeoutError("Task: %d Timeout" % self._id)

        # remove from future_cache
        future_cache = self._future_cache_ref()
        if future_cache is not None:
            del future_cache[self._id]

        # [(request_id, output), ...] sorted by request_id
        self._outputs.sort(key=lambda i:i[0])
        # restore batch result from outputs
        batch_result = [i[1] for i in self._outputs]

        return batch_result

    def done(self):
        if self._finish_event.is_set():
            return True

    def _append_result(self, it_id, it_output):
        self._outputs.append((it_id, it_output))
        if len(self._outputs) >= self._size:
            self._finish_event.set()
            print("%d task_id:%d size:%d finished" % (os.getpid(), self._id, self._size))


class _FutureCache(dict):
    "Dict for weakref only"
    pass


class _BaseStreamer(object):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._client_id = str(uuid.uuid4())
        self._task_id = 0
        self._future_cache = _FutureCache()  # {task_id: future}

        self.back_thread = threading.Thread(target=self._loop_collect_result, name="thread_collect_result")
        self.back_thread.daemon = True

    def _delay_setup(self):
        self.back_thread.start()

    def _send_request(self, task_id, request_id, model_input):
        raise NotImplementedError

    def _recv_response(self, timeout=1):
        raise NotImplementedError

    def _input(self, batch: List) -> int:
        """
        input a batch, distribute each item to mq, return task_id
        """
        # task id in one client
        task_id = self._task_id
        self._task_id += 1
        # request id in one task
        request_id = 0

        for model_input in batch:
            self._send_request(task_id, request_id, model_input)
            print("_send_request", os.getpid(), task_id, request_id)
            print("enqueue", os.getpid(), task_id, request_id)
            request_id += 1

        future = Future(task_id, request_id, weakref.ref(self._future_cache))
        self._future_cache[task_id] = future

        return task_id

    def _loop_collect_result(self):
        print(self, "start _loop_collect_result")
        while True:
            message = self._recv_response(timeout=1)
            if message:
                (task_id, request_id, item) = message
                future = self._future_cache[task_id]
                future._append_result(request_id, item)
            else:
                # todo
                time.sleep(0.001)

    def _output(self, task_id: int) -> List:
        future = self._future_cache[task_id]
        batch_result = future.result(20)  # 20s timeout for any requests
        return batch_result

    def submit(self, batch):
        task_id = self._input(batch)
        future = self._future_cache[task_id]
        return future

    def predict(self, batch):
        task_id = self._input(batch)
        ret = self._output(task_id)
        return ret


class _BaseStreamWorker(object):
    def __init__(self, predict_function, batch_size, max_latency, *args, **kwargs):
        super().__init__()
        assert callable(predict_function)
        self._predict = predict_function
        self._batch_size = batch_size
        self._max_latency = max_latency

    def run_forever(self):
        print(self, "start working")

        while True:
            handled = self._run_once()
            if not handled:
                # sleep if no data handled last time
                time.sleep(0.001)

    def model_predict(self, batch_input):
        # fairseq (gpu)
        batch_result : List[str] = self._predict(batch_input)
        return batch_result

    def _run_once(self):
        batch = []
        start_time = time.time()
        for i in range(self._batch_size):
            try:
                item = self._recv_request(timeout=self._max_latency)
            except TimeoutError:
                # each item timeout exceed the max latency
                break
            else:
                batch.append(item)
            if (time.time() - start_time) > self._max_latency:
                # total batch time exceeds the max latency
                break
        if not batch:
            return 0

        model_inputs = [i[3] for i in batch]
        model_outputs = self.model_predict(model_inputs)

        # publish results to redis
        for i, item in enumerate(batch):
            client_id, task_id, request_id, _ = item
            self._send_response(client_id, task_id, request_id, model_outputs[i])

        batch_size = len(batch)
        print("run_once batch_size: %d start_at: %s spend: %s" % (batch_size, start_time, time.time() - start_time))
        return batch_size

    def _recv_request(self, timeout=1):
        raise NotImplementedError

    def _send_response(self, client_id, task_id, request_id, model_input):
        raise NotImplementedError


class ThreadedStreamer(_BaseStreamer):
    def __init__(self, predict_function, batch_size, max_latency=0.1):
        super().__init__()
        self._input_queue = Queue()
        self._output_queue = Queue()
        self._worker = ThreadedWorker(predict_function, batch_size, max_latency, self._input_queue, self._output_queue)
        self._worker_thread = threading.Thread(target=self._worker.run_forever, name="thread_worker")
        self._worker_thread.daemon = True
        self._worker_thread.start()
        self._delay_setup()

    def _send_request(self, task_id, request_id, model_input):
        self._input_queue.put((0, task_id, request_id, model_input))

    def _recv_response(self, timeout=1):
        try:
            message = self._output_queue.get(timeout=1)
        except Empty:
            message = None
        return message


class ThreadedWorker(_BaseStreamWorker):
    def __init__(self, predict_function, batch_size, max_latency, request_queue, response_queue):
        super().__init__(predict_function, batch_size, max_latency)
        self._request_queue = request_queue
        self._response_queue = response_queue

    def _recv_request(self, timeout=1):
        try:
            item = self._request_queue.get(timeout=self._max_latency)
        except Empty:
            raise TimeoutError
        else:
            return item

    def _send_response(self, client_id, task_id, request_id, model_output):
        self._response_queue.put((task_id, request_id, model_output))


class Streamer(_BaseStreamer):
    """
    1. input batch as a task
    2. distribute every single item in batch to redis
    3. backend loop collecting results
    3. output batch result for a task when every single item is returned
    """
    def __init__(self):
        super().__init__()
        self._redis = _RedisClient(self._client_id)
        self._delay_setup()

    def _send_request(self, task_id, request_id, model_input):
        self._redis.send_request(task_id, request_id, model_input)

    def _recv_response(self, timeout=1):
        return self._redis.recv_response(timeout)


class StreamWorker(_BaseStreamWorker):
    def __init__(self, predict_function, batch_size, max_latency=0.1):
        super().__init__(predict_function, batch_size, max_latency)
        self._redis = _RedisServer(0)
        self._requests_queue = Queue()

        self.back_thread = threading.Thread(target=self._loop_recv_request, name="thread_recv_request")
        self.back_thread.daemon = True
        self.back_thread.start()

    def _loop_recv_request(self):
        print(self, "start loop_recv_request")
        while True:
            message = self._redis.recv_request(1)
            if message:
                (client_id, task_id, request_id, request_item) = json.loads(message)
                self._requests_queue.put((client_id, task_id, request_id, request_item))
            else:
                # sleep if recv timeout
                time.sleep(0.001)

    def _recv_request(self, timeout=1):
        try:
            item = self._requests_queue.get(timeout=self._max_latency)
        except Empty:
            raise TimeoutError
        else:
            return item

    def _send_response(self, client_id, task_id, request_id, model_output):
        self._redis.send_response(client_id, task_id, request_id, model_output)


class _RedisAgent(object):
    def __init__(self, redis_id):
        self._redis_id = redis_id
        self._redis_request_queue_name = "request_queue"
        self._redis_response_pb_prefix = "response_pb_"
        self._redis = Redis()
        self._response_pb = self._redis.pubsub(ignore_subscribe_messages=True)
        self._setup()

    def _setup(self):
        raise NotImplementedError

    def _response_pb_name(self, redis_id):
        return self._redis_response_pb_prefix + redis_id


class _RedisClient(_RedisAgent):
    def _setup(self):
        self._response_pb.subscribe(self._response_pb_name(self._redis_id))

    def send_request(self, task_id, request_id, model_input):
        message = (self._redis_id, task_id, request_id, model_input)
        self._redis.lpush(self._redis_request_queue_name, json.dumps(message))

    def recv_response(self, timeout):
        message = self._response_pb.get_message(timeout=timeout)
        if message:
            return json.loads(message["data"])


class _RedisServer(_RedisAgent):
    def _setup(self):
        # server subscribe all pubsub
        self._response_pb.psubscribe(self._redis_response_pb_prefix + "*")

    def recv_request(self, timeout):
        message = self._redis.blpop(self._redis_request_queue_name, timeout=timeout)
        # (queue_name, data)
        if message:
            return message[1]

    def send_response(self, client_id, task_id, request_id, model_output):
        message = (task_id, request_id, model_output)
        channel_name = self._response_pb_name(client_id)
        self._redis.publish(channel_name, json.dumps(message))