# TensorDictBase.isnan

TensorDictBase.isnan() → Self

Returns a new tensordict with boolean elements representing if each element of input is NaN or not.

Complex values are considered NaN when either their real and/or imaginary part is NaN.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).