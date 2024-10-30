import easyocr
import numpy as np
import cv2

class Reader:
    def __init__(self, lang_list: list[str] = ['en']) -> None:
        self._reader: easyocr.Reader = easyocr.Reader(lang_list)
        self.outputs: list[tuple[list, str, float]]  = []
        self.image: np.ndarray = np.ndarray(0)


    def read(self, img: np.ndarray, verbosity: int | None=None) -> list[tuple[list, str, float]]:
        
        self.image = img
        self.outputs = self._reader.readtext(img)
        
        if verbosity is not None:
            if verbosity == 0:
                for _, t, _ in self.outputs:
                    print(t)
            elif verbosity == 1:
                for _, t, s in self.outputs:
                    print(t, int(s*100))
            elif verbosity == 2:
                for b, t, s in self.outputs:
                    print(b, t, int(s*100))

        return self.outputs


    def text(self) -> list[str]:
        return [t for _, t, _ in self.outputs]


    def boxes(self) -> list:
        return [b for b, _, _ in self.outputs]
    

    def scores(self) -> list[float]:
        return [s for _, _, s in self.outputs]
    
    
    def draw_predictions(self, stop: bool=True):
        for output in self.outputs:
            bbox, text, score = output
            bbox = np.asarray(bbox).astype(int)
            cv2.rectangle(self.image, bbox[0], bbox[2], (0,255,0), 2)
            cv2.putText(self.image, f'{text} {score*100:.1f}%', bbox[0], cv2.FONT_HERSHEY_COMPLEX, .5, (255,0,0), 1)
            
        cv2.imshow('prediction', self.image)
        
        if stop:
            cv2.waitKey(0)
        