from pathlib import Path
from typing import Union

import numpy
from asm.api.ai import ASMAI, AIResult, AIExpansion
from asm.api.base import ModuleInformation, ModuleTask, ModuleTaskInput, ModuleTaskOutput, ModuleConfiguration, \
    ContainerParameterResults


class TFLite(ASMAI):
    def expansions(self) -> AIExpansion:
        pass

    def available_labels(self, labels: Path) -> list[str]:
        pass

    def process(self, frame: numpy.ndarray) -> tuple[AIResult, Union[list[ContainerParameterResults], None]]:
        pass

    def load(self, model: Path, labels: Path) -> bool:
        pass

    def unload(self) -> None:
        pass

    def module_info(self) -> ModuleInformation:
        pass

    def configuration(self, configuration: ModuleConfiguration):
        pass

    def task(self, task: ModuleTask, task_input: ModuleTaskInput) -> Union[ModuleTaskOutput, None]:
        pass