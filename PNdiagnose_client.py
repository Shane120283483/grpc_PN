"""The Python implementation of the GRPC PNdiagnose.Diagnoser client."""

from __future__ import print_function

import logging

import grpc
import PNdiagnose_pb2
import PNdiagnose_pb2_grpc
import base64
import numpy as np

test_id0 = "214d0a16-32c44836-78e235b9-5c429059-8f9f5c57"
test_id1 = "c20b088c-1b28f2b6-c31bc0e3-9b659f75-7a820d23"
test_id2 = "12b6ae41-ef40293c-9999b1ac-bc06d52a-0520033e"
test_id3 = ""  # 1001 url错误
test_id4 = "https://geekuninstaller.com/geek.zip"  # 1002 zip内没有.dcm
test_id5 = "https://dldir1.qq.com/qqfile/qq/PCQQ9.5.9/QQ9.5.9.28650.exe"  # 1001 非zip


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = PNdiagnose_pb2_grpc.DiagnoserStub(channel)
        response = stub.GetDiagnosis(
            PNdiagnose_pb2.DiagRequest(Identifier=test_id0))
        print("Diagnoser client received: %s, code:%s" % (str(response.type), str(response.code)))
        received_cam = base64.b64decode(response.cam_img)
        # print(received_cam)
        imgarr = np.frombuffer(received_cam, dtype=np.float32)
        print(imgarr)
        cam_img = imgarr.reshape((69, 95, 79))
        print(cam_img)
        print(type(cam_img))


if __name__ == '__main__':
    logging.basicConfig()
    run()
