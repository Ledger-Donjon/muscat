# muscatpy
This package provides python bindings to [muscat](https://github.com/Ledger-Donjon/muscat).

## Build dependencies
- [rust](https://www.rust-lang.org/tools/install)
- [maturin](https://github.com/PyO3/maturin)

## Install
Build locally and install the wheel:
```sh
maturin build --release
# The name of the produced wheel depends on your python toolchain
pip install target/wheels/muscatpy-*.whl
```

Or install it from the git repository:
```sh
pip install 'git+https://github.com/Ledger-Donjon/muscat.git#egg=muscatpy&subdirectory=muscatpy'
```

## License
Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution
Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
