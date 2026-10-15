# TensorDictBase.isfinite

TensorDictBase.isfinite() → Self

Returns a new tensordict with boolean elements representing if each element is finite or not.

Real values are finite when they are not NaN, negative infinity, or infinity. Complex values are finite when both their real and imaginary parts are finite.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).