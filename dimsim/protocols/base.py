"""
Base Protocol class.
"""
import abc
import typing

from ..base import BaseModel
from ..configs.protocols import ProtocolConfig

if typing.TYPE_CHECKING:
    from ..workflow.task import Task

class Protocol(BaseModel, abc.ABC):
    """An abstract base class for all protocols to inherit from"""

    name: typing.ClassVar[str]
    config: ProtocolConfig

    @classmethod
    def _get_execution_outputs(cls) -> tuple[str, ...]:
        """Return the names of the outputs produced by this protocol"""
        return tuple()

    @abc.abstractmethod
    def _execute(
        self,
        **kwargs
    ) -> typing.Any:
        """Execute the protocol and return the outputs as a tuple"""
        pass


    def execute_on_task(self, task: "Task") -> "Task":
        """
        Execute the protocol on a given task and return the updated task

        # TODO: it may not be useful to separate this from _execute
        and fuss around with _get_execution_outputs.
        # Mainly this separation exists so that Protocols can be used
        both within a Workflow and independently, which e.g. helps
        with testing and debugging.
        """
        inputs = dict(task.inputs)
        outputs = self._execute(**inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        output_names = self._get_execution_outputs()
        outputs = dict(task.outputs)
        for name, value in zip(output_names, outputs):
            if name in outputs:
                if not isinstance(outputs[name], type(value)):
                    raise ValueError(
                        f"Expected type {type(outputs[name])} for output "
                        f"'{name}', but got type {type(value)}"
                    )
            outputs[name] = value

        task.outputs = outputs
        return task
