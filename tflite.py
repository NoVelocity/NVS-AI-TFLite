from pathlib import Path
from typing import Union

import cv2
import numpy
import tensorflow
from asm.api.ai import ASMAI, AIResult, AIExpansion
from asm.api.base import ModuleInformation, ModuleTask, ModuleTaskInput, ModuleTaskOutput, ModuleConfiguration, \
    ContainerParameterResults, ModuleRequirement


class TFLite(ASMAI):
    interpreter = None
    input_details = None
    output_details = None
    current_labels = None

    model_name: str = ""

    def expansions(self) -> AIExpansion:
        return AIExpansion(["txt"], ["tflite"])

    def available_labels(self, labels: Path) -> list[str]:
        return self.current_labels

    def process(self, frame: numpy.ndarray) -> tuple[Union[AIResult, None], Union[list[ContainerParameterResults], None]]:
        if self.interpreter is None:
            raise ModuleNotFoundError()

        h, w = self.input_details[0]["shape"][1:3]

        img = cv2.resize(frame, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = numpy.expand_dims(img, axis=0)

        if self.input_details[0]["dtype"] == numpy.float32:
            img = img.astype(numpy.float32) / 255.0

        self.interpreter.set_tensor(self.input_details[0]["index"], img)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        class_id = numpy.argmax(output)

        return AIResult(self.model_name, self.current_labels[class_id]), None

    def load(self, model: Path, labels: Path) -> bool:
        self.model_name = model.name

        self.interpreter = tensorflow.lite.Interpreter(model_path=model)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.current_labels = []
        with open(labels, "r") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                self.current_labels.append(parts[1])

        return True

    def unload(self) -> None:
        self.model_name = ""
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.current_labels = None

    def module_info(self) -> ModuleInformation:
        return ModuleInformation(
            name="AI-TFLite",
            version="1.0.0",
            requirements=[ModuleRequirement("tensorflow")]
        )

    def configuration(self, configuration: ModuleConfiguration):
        return None

    def task(self, task: ModuleTask, task_input: ModuleTaskInput) -> Union[ModuleTaskOutput, None]:
        return None
