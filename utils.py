import numpy as np
import depthai as dai

# TODO
class Message:
    def __init__(self, content: list[str]) -> None:
        self.content = content
    

class CTCCodec(object):
	""" Convert between text-label and text-index """
	def __init__(self, characters):
		# characters (str): set of the possible characters.
		dict_character = list(characters)

		self.dict = {}
		for i, char in enumerate(dict_character):
			self.dict[char] = i + 1

		self.characters = dict_character


	def decode(self, preds):
		""" convert text-index into text-label. """
		texts = []
		index = 0
		# Select max probabilty (greedy decoding) then decode index to character
		preds = preds.astype(np.float16)
		preds_index = np.argmax(preds, 2)
		preds_index = preds_index.transpose(1, 0)
		preds_index_reshape = preds_index.reshape(-1)
		preds_sizes = np.array([preds_index.shape[1]] * preds_index.shape[0])

		for l in preds_sizes:
			t = preds_index_reshape[index:index + l]

			# NOTE: t might be zero size
			if t.shape[0] == 0:
				continue

			char_list = []
			for i in range(l):
				# removing repeated characters and blank.
				if not (i > 0 and t[i - 1] == t[i]):
					if self.characters[t[i]] != '#':
						char_list.append(self.characters[t[i]])
			text = ''.join(char_list)
			texts.append(text)

			index += l

		return texts
	

class HostSeqSync:
    def __init__(self):
        self.imgFrames = []
    def add_msg(self, msg):
        self.imgFrames.append(msg)
    def get_msg(self, target_seq):
        for i, imgFrame in enumerate(self.imgFrames):
            if target_seq == imgFrame.getSequenceNum():
                self.imgFrames = self.imgFrames[i:]
                break
        return self.imgFrames[0]
	

def create_pipeline():
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_2)

    colorCam = pipeline.create(dai.node.ColorCamera)
    colorCam.setPreviewSize(256, 256)
    colorCam.setVideoSize(1024, 1024) # 4 times larger in both axis
    colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    colorCam.setInterleaved(False)
    colorCam.setBoardSocket(dai.CameraBoardSocket.RGB)
    colorCam.setFps(10)

    controlIn = pipeline.create(dai.node.XLinkIn)
    controlIn.setStreamName('control')
    controlIn.out.link(colorCam.inputControl)

    cam_xout = pipeline.create(dai.node.XLinkOut)
    cam_xout.setStreamName('video')
    colorCam.video.link(cam_xout.input)

    # ---------------------------------------
    # 1st stage NN - text-detection
    # ---------------------------------------

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath('.\\models\\east_text_detection_256x256_openvino_2021.2_6shave.blob')
    colorCam.preview.link(nn.input)

    nn_xout = pipeline.create(dai.node.XLinkOut)
    nn_xout.setStreamName('detections')
    nn.out.link(nn_xout.input)

    # ---------------------------------------
    # 2nd stage NN - text-recognition-0012
    # ---------------------------------------

    manip = pipeline.create(dai.node.ImageManip)
    manip.setWaitForConfigInput(True)

    manip_img = pipeline.create(dai.node.XLinkIn)
    manip_img.setStreamName('manip_img')
    manip_img.out.link(manip.inputImage)

    manip_cfg = pipeline.create(dai.node.XLinkIn)
    manip_cfg.setStreamName('manip_cfg')
    manip_cfg.out.link(manip.inputConfig)

    manip_xout = pipeline.create(dai.node.XLinkOut)
    manip_xout.setStreamName('manip_out')

    nn2 = pipeline.create(dai.node.NeuralNetwork)
    nn2.setBlobPath('.\\models\\text-recognition-0012_openvino_2021.2_6shave.blob')
    nn2.setNumInferenceThreads(2)
    manip.out.link(nn2.input)
    manip.out.link(manip_xout.input)

    nn2_xout = pipeline.create(dai.node.XLinkOut)
    nn2_xout.setStreamName("recognitions")
    nn2.out.link(nn2_xout.input)

    return pipeline


def to_tensor_result(packet):
    return {
        name: np.array(packet.getLayerFp16(name))
        for name in [tensor.name for tensor in packet.getRaw().tensors]
    }


def to_planar(frame):
    return frame.transpose(2, 0, 1).flatten()

def camera_settings():
    pass