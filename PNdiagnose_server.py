"""The Python implementation of the GRPC PNdiagnose.Diagnoser server."""
import os
from concurrent import futures
import logging
import grpc
import PNdiagnose_pb2
import PNdiagnose_pb2_grpc
import classifier
import base64

archive_url = "http://27.17.30.150:20086/series/ID/archive"

test_id0 = "214d0a16-32c44836-78e235b9-5c429059-8f9f5c57"
test_id1 = "c20b088c-1b28f2b6-c31bc0e3-9b659f75-7a820d23"
test_id2 = "12b6ae41-ef40293c-9999b1ac-bc06d52a-0520033e"
test_id3 = ""  # 1001

ERROR_ID_URL = 1001
ERROR_NO_DCM_DIR = 1002


class DiagnoserServicer(PNdiagnose_pb2_grpc.DiagnoserServicer):
    def getDCM2NII(self, request, context):
        cur_id = request.Identifier
        cur_dcm_url = archive_url.replace("ID", cur_id)
        # cur_dcm_url = cur_id
        dcm_dir_path = classifier.unzip_dcm(cur_dcm_url)  # 获取url对应zip，解压到本地，获取.dcm对应父目录
        print(dcm_dir_path)
        if dcm_dir_path is not None and dcm_dir_path != "":
            data_path = "unzip_data/cur.nii"
            classifier.dcm2nii(dcm_dir_path, data_path)
        return dcm_dir_path

    def getPred(self, request, context):
        dcm_dir_path = self.getDCM2NII(request, context)
        data_path = "unzip_data/cur.nii"
        if dcm_dir_path is None:
            return ERROR_ID_URL
        elif dcm_dir_path == "":
            return ERROR_NO_DCM_DIR
        return classifier.predict(data_path)  # class

    def getPredCam(self, request, context):
        dcm_dir_path = self.getDCM2NII(request, context)
        data_path = "unzip_data/cur.nii"
        if dcm_dir_path is None:
            return ERROR_ID_URL
        elif dcm_dir_path == "":
            return ERROR_NO_DCM_DIR
        return classifier.show_CAM(data_path)  # class, cam_img

    def GetDiagnosis(self, request, context):
        # pred_type = self.getPred(request, context)
        pred_type, cam_img = self.getPredCam(request, context)
        print(pred_type)
        print(cam_img)
        print(cam_img.shape)
        if pred_type == ERROR_ID_URL or pred_type == ERROR_NO_DCM_DIR:
            return PNdiagnose_pb2.DiagReply(type=pred_type, cam_img=str(pred_type), code=pred_type)
        else:
            return PNdiagnose_pb2.DiagReply(type=pred_type, cam_img=base64.b64encode(cam_img), code=200)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    PNdiagnose_pb2_grpc.add_DiagnoserServicer_to_server(DiagnoserServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
    # cur_dcm_url = archive_url.replace("ID", test_id1)
    # dcm_dir_path = classifier.unzip_dcm(cur_dcm_url)  # 获取url对应zip，解压到本地，获取.dcm对应父目录
    # classifier.dcm2nii(dcm_dir_path, "unzip_data/cur.nii")
