import abc
import typing


class Backend(abc.ABC):

    @abc.abstractmethod
    def __init__(self, host: str, port: int):
        """
        Initialize the backend with the given host and port.

        Parameters
        ----------
        host : str
            The host address of the backend.
        port : int
            The port number of the backend.
        """
        pass

    @abc.abstractmethod
    def submit(self, func: typing.Callable, *args, **kwargs) -> typing.Any:
        """
        Submit a function to be executed in the backend.

        Parameters
        ----------
        func : typing.Callable
            The function to execute.
        *args
            Positional arguments to pass to the function.
        **kwargs
            Keyword arguments to pass to the function.

        Returns
        -------
        typing.Any
            A future or result of the execution.
        """
        pass

    @abc.abstractmethod
    def as_completed(
        self,
        futures: typing.List[typing.Any],
        chunksize: int = 1,
        return_results: bool = False,
        return_futures: bool = True,
    ) -> typing.Iterator[typing.List[typing.Any]]:
        """
        Iterate over futures as they complete.

        Parameters
        ----------
        futures : typing.List[typing.Any]
            A list of futures to monitor.
        chunksize : int, optional
            The number of completed futures to yield at once, by default 1.
        return_results : bool, optional
            Whether to return results instead of futures, by default False.
        return_futures : bool, optional
            Whether to return futures instead of results, by default True.

        Returns
        -------
        typing.Iterator[typing.List[typing.Any]]
            An iterator over lists of completed futures.
            If `return_futures` is True and `return_results` is False, yields lists of futures.
            If `return_results` is True and `return_futures` is False, yields lists of results.
            If both are True, yields lists of tuples (future, result).
        """
        pass


class DaskBackend(Backend):
    """
    A Dask backend for clients.
    """

    backend: typing.ClassVar[typing.Literal["dask"]] = "dask"


    def __init__(self, host: str = "localhost", port: int = 8786):
        from dask.distributed import Client as DaskClient

        self._host = host
        self._port = port
        self._client = DaskClient(f"{host}:{port}")

    @property
    def address(self) -> str:
        return f"{self._host}:{self._port}"

    def scatter(self, data: typing.Any, broadcast: bool = True) -> typing.Any:
        """
        Scatter data to the Dask cluster.

        Parameters
        ----------
        data : typing.Any
            The data to scatter.
        broadcast : bool, optional
            Whether to broadcast the data to all workers, by default True.

        Returns
        -------
        typing.Any
            A future representing the scattered data.
        """
        return self._client.scatter(data, broadcast=broadcast)

    def as_completed(
        self,
        futures: typing.List[typing.Any],
        chunksize: int = 1,
        return_results: bool = False,
        return_futures: bool = True,
    ) -> typing.Iterator[typing.List[typing.Any]]:
        """
        Iterate over futures as they complete.

        Parameters
        ----------
        futures : typing.List[typing.Any]
            A list of futures to monitor.
        chunksize : int, optional
            The number of completed futures to yield at once, by default 1.
        return_results : bool, optional
            Whether to return results instead of futures, by default False.
        return_futures : bool, optional
            Whether to return futures instead of results, by default True.

        Returns
        -------
        typing.Iterator[typing.List[typing.Any]]
            An iterator over lists of completed futures.
            If `return_futures` is True and `return_results` is False, yields lists of futures.
            If `return_results` is True and `return_futures` is False, yields lists of results.
            If both are True, yields lists of tuples (future, result).
        """
        from dask.distributed import as_completed

        assert return_results or return_futures, "Must return either results or futures"

        ac = as_completed(futures, with_results=return_results)
        chunks: list = []
        for future in ac:
            chunks.append(future)
            if len(chunks) >= chunksize:
                if return_results and not return_futures:
                    chunks = [result for _, result in chunks]
                yield chunks
                chunks = []
        if chunks:
            if return_results and not return_futures:
                chunks = [result for _, result in chunks]
            yield chunks

    def gather(self, futures: typing.List[typing.Any]) -> typing.List[typing.Any]:
        """
        Gather results from futures.

        Parameters
        ----------
        futures : typing.List[typing.Any]
            A list of futures to gather results from.

        Returns
        -------
        typing.List[typing.Any]
            A list of results corresponding to the input futures.
        """
        return self._client.gather(futures)


class Client:
    """
    A client to manage distributed computation.

    Initially this will just wrap Dask,
    but it is written to be extensible to other backends in the future.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8786,
        backend: typing.Literal["dask"] = "dask",
    ):
        self._host = host
        self._port = port
        if backend == "dask":
            self._backend = DaskBackend(host, port)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    @property
    def backend(self) -> Backend:
        return self._backend

    def scatter(self, data: typing.Any, broadcast: bool = False) -> typing.Any:
        """
        Scatter data to the backend.

        Parameters
        ----------
        data : typing.Any
            The data to scatter.
        broadcast : bool, optional
            Whether to broadcast the data to all workers, by default False.

        Returns
        -------
        typing.Any
            A future representing the scattered data.
        """
        return self._backend.scatter(data, broadcast=broadcast)

    def as_completed(
        self,
        futures: typing.List[typing.Any],
        chunksize: int = 1,
        return_results: bool = False,
        return_futures: bool = True,
    ) -> typing.Iterator[typing.List[typing.Any]]:
        """
        Iterate over futures as they complete.
        Note we normally expect to want the results rather than the futures themselves.
        We set the defaults to match the identically named Dask function.

        Parameters
        ----------
        futures : typing.List[typing.Any]
            A list of futures to monitor.
        chunksize : int, optional
            The number of completed futures to yield at once, by default 1.
        return_results : bool, optional
            Whether to return results instead of futures, by default False.
        return_futures : bool, optional
            Whether to return futures instead of results, by default True.

        Returns
        -------
        typing.Iterator[typing.List[typing.Any]]
            An iterator over lists of completed futures.
            If `return_futures` is True and `return_results` is False, yields lists of futures.
            If `return_results` is True and `return_futures` is False, yields lists of results.
            If both are True, yields lists of tuples (future, result).
        """
        for chunk in self._backend.as_completed(
            futures,
            chunksize=chunksize,
            return_results=return_results,
            return_futures=return_futures,
        ):
            yield chunk


    def submit(self, func: typing.Callable, *args, **kwargs) -> typing.Any:
        """
        Submit a function to be executed in the backend.

        Parameters
        ----------
        func : typing.Callable
            The function to execute.
        *args
            Positional arguments to pass to the function.
        **kwargs
            Keyword arguments to pass to the function.

        Returns
        -------
        typing.Any
            A future or result of the execution.
        """
        return self._backend.submit(func, *args, **kwargs)

    def gather(self, futures: typing.List[typing.Any]) -> typing.List[typing.Any]:
        """
        Gather results from futures.

        Parameters
        ----------
        futures : typing.List[typing.Any]
            A list of futures to gather results from.

        Returns
        -------
        typing.List[typing.Any]
            A list of results corresponding to the input futures.
        """
        return self._backend.gather(futures)
