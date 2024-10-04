'''
A collection of helper functions for functional programming.
'''

from typing import Sequence, Any, Callable, TypeVar, Generic, Union
from collections.abc import Sequence as TypeofSequence

'''
Generic type variables.
'''
T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')



def tap(f: callable):
    '''
    Performs a side effect on the input and returns the input.
    
    Parameters
    ----------
    f : callable
        The function to be applied to the input.
    
    Returns
    -------
    callable
        A function that applies the input function to the input and returns the input.
    '''
    def wrapper(x):
        f(x)
        return x
    return wrapper

def is_sequence(x: Any) -> bool:
    '''
    Checks if the input is a sequence.
    
    Parameters
    ----------
    x : any
        The input to be checked.
    
    Returns
    -------
    bool
        True if the input is a sequence, False otherwise.
    '''
    return isinstance(x, TypeofSequence) and not isinstance(x, str) and not isinstance(x, bytes) and not isinstance(x, bytearray)

def flatten(arr: Sequence[Sequence[T]], depth: int =-1):
    """
    Flattens a nested list up to a specified depth.

    Parameters:
    -----------
    arr Sequence[Sequence[T]]:
        The Sequence to be flattened.

    depth (int, optional): 
        The depth to which the list should be flattened. 
        If -1, the list is flattened completely. Defaults to -1.
    
    Returns:
    list: 
        A flattened list up to the specified depth.
    """

    # Base case
    if depth == 0 or not is_sequence(arr):
        return arr
    
    flattened_list = []
    for item in arr:
        # If the item is a sequence and we're not at our depth limit, extend by flattened item, else append item directly
        if is_sequence(item) and depth != 1:
            flattened_list.extend(flatten(item, depth - 1))
        else:
            flattened_list.append(item)
    
    return flattened_list

class Maybe(Generic[T]):
    def __init__(self, V: T = None) -> None:
        '''
        Creates a Maybe object.
        
        Parameters
        ----------
        V : T, optional (default=None)
            The value to be wrapped in the Maybe.
        '''
        self._value = V
        
    def flat_map(self, f: Callable[..., 'Maybe[T]']) -> 'Maybe[T]':
        '''
        Binds a function that returns a Maybe to a Maybe.
        
        Parameters
        ----------
        f : Callable[..., 'Maybe[T]']
            The function to be bound to the Maybe.
        
        Returns
        -------
        Maybe[T]
            A Maybe object.
        '''
        return self if self.empty() else f(self._value) # apply f to the value and return the resulting Maybe
        
    @staticmethod
    def unit(V: T = None) -> 'Maybe[T]':
        '''
        Unit function for Maybe monad.
        Creates a Maybe object.
        This is equivalent to Maybe(V, default) and is provided for consistency with the Maybe monad.
        
        Parameters
        ----------
        V : T
            The value to be wrapped in the Maybe.
        '''
        return Maybe(V)
    
    @staticmethod
    def empty(cls, maybe) -> bool:
        '''
        Checks if the Maybe is empty.
        
        Returns
        -------
        bool
            True if the Maybe is empty, False otherwise.
        '''
        return maybe._value is None
    
    @staticmethod
    def value_or_else(cls, maybe: 'Maybe[T]', default: T) -> T:
        '''
        Makes the Maybe certain by returning the value if it exists, or the default value otherwise.
        
        Parameters
        ----------
        maybe : Maybe[T]
            The Maybe to be unwrapped.
        default : T
            The default value to be returned if the Maybe is empty.
        
        Returns
        -------
        T
            The value of the Maybe if it exists, or the default value otherwise.
        '''
        return maybe._value if maybe._value is not None else default
        
    def __str__(self) -> str:
        '''
        Returns a string representation of the Maybe object.
        '''
        return f'Maybe({self._value})'
    

class Either(Generic[T, U]):
    def __init__(self, V: Union[T, U], is_right: bool) -> None:
        '''
        Creates an Either object.
        
        Parameters
        ----------
        V : Union[T, U] 
            The value to be wrapped in the Either.
        '''
        self._value = V
        self._is_right = is_right
        
    def is_left(self) -> bool:
        '''
        Checks if the Either is a Left.
        
        Returns
        -------
        bool
            True if the Either is a Left, False otherwise.
        '''
        return not self._is_right
    
    def is_right(self) -> bool:
        '''
        Checks if the Either is a Right.
        
        Returns
        -------
        bool
            True if the Either is a Right, False otherwise.
        '''
        return self._is_right

        
    def flat_map(self, f: Callable[..., 'Either[T, U]']) -> 'Either[T, U]':
        '''
        Binds a function that returns an Either to an Either.
        
        Parameters
        ----------
        f : Callable[..., 'Either[T, U]']
            The function to be bound to the Either.
        
        Returns
        -------
        Either[T, U]
            An Either object.
        '''
        return self if self.is_left() else f(self._unpack()) # apply f to the value and return the resulting Either
        
    
    @staticmethod
    def unit(V: Union[T, U], is_right: bool) -> 'Either[T, U]':
        '''
        Unit function for Either monad.
        Creates an Either object.
        This is equivalent to Either(V, default) and is provided for consistency with the Either monad.
        
        Parameters
        ----------
        V : Union[T, U]
            The value to be wrapped in the Either.
        '''
        return Either(V, is_right)
    
    @staticmethod
    def match(either: 'Either[T, U]', left: Callable[[T], V], right: Callable[[U], V]) -> V:
        return left(either._value) if either.is_left() else right(either._value)
    
    @staticmethod
    def wrap_exception(f: Callable[..., T]) -> Callable[..., 'Either[T, Exception]']:
        def wrapper(*args, **kwargs) -> Either[T, Exception]:
            try:
                return Either(f(*args, **kwargs), is_right=True)
            except Exception as e:
                return Either(e, is_right=False)
        return wrapper
        
    def __str__(self) -> str:
        '''
        Returns a string representation of the Either object.
        '''
        return f'Either({self._value}, {self._is_right})'
    
    