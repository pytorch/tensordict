window.BENCHMARK_DATA = {
  "lastUpdate": 1792423878219,
  "repoUrl": "https://github.com/pytorch/tensordict",
  "entries": {
    "CPU Benchmark Results": [
      {
        "commit": {
          "author": {
            "email": "peterdsharpe@gmail.com",
            "name": "Peter Sharpe",
            "username": "peterdsharpe"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "687b27a6113d08ba5fb141b85b573c1593abab65",
          "message": "[CI] Keep 8 days of benchmark charts instead of 3, and stop timing CPU-only benchmarks in the GPU job (#3058)",
          "timestamp": "2026-10-19T10:57:24-04:00",
          "tree_id": "3e44f3cdb322968149c2db6f09ca8de53f253a07",
          "url": "https://github.com/pytorch/tensordict/commit/687b27a6113d08ba5fb141b85b573c1593abab65"
        },
        "date": 1792423583014,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cpu-32-2]",
            "value": 14400.4772889,
            "unit": "iter/sec",
            "range": "stddev: 0.00000247694236152",
            "extra": "mean: 69.4421427801 usec\nrounds: 3866"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cpu-32-8]",
            "value": 9565.30005499,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027814669755",
            "extra": "mean: 104.54455106 usec\nrounds: 5571"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cpu-32-32]",
            "value": 3845.2915018,
            "unit": "iter/sec",
            "range": "stddev: 0.00000446962570414",
            "extra": "mean: 260.058307551 usec\nrounds: 2611"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cpu-4096-2]",
            "value": 5153.76166536,
            "unit": "iter/sec",
            "range": "stddev: 0.0000176689871434",
            "extra": "mean: 194.033031586 usec\nrounds: 1172"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cpu-4096-8]",
            "value": 1736.80550999,
            "unit": "iter/sec",
            "range": "stddev: 0.0000247993119141",
            "extra": "mean: 575.76970723 usec\nrounds: 386"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cpu-4096-32]",
            "value": 352.620984768,
            "unit": "iter/sec",
            "range": "stddev: 0.0000525152709594",
            "extra": "mean: 2.8359060952 msec\nrounds: 126"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cpu-32-2]",
            "value": 7551.07283084,
            "unit": "iter/sec",
            "range": "stddev: 0.00000338651227591",
            "extra": "mean: 132.431513032 usec\nrounds: 2497"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cpu-32-8]",
            "value": 1929.27534678,
            "unit": "iter/sec",
            "range": "stddev: 0.00000843150305769",
            "extra": "mean: 518.329331099 usec\nrounds: 1655"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cpu-32-32]",
            "value": 478.313155091,
            "unit": "iter/sec",
            "range": "stddev: 0.000016537953005",
            "extra": "mean: 2.09068052876 msec\nrounds: 452"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cpu-4096-2]",
            "value": 4106.8687734,
            "unit": "iter/sec",
            "range": "stddev: 0.0000374784007441",
            "extra": "mean: 243.494510094 usec\nrounds: 2041"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cpu-4096-8]",
            "value": 967.612800838,
            "unit": "iter/sec",
            "range": "stddev: 0.0000319439914721",
            "extra": "mean: 1.03347123884 msec\nrounds: 360"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cpu-4096-32]",
            "value": 194.748555191,
            "unit": "iter/sec",
            "range": "stddev: 0.0000566823618219",
            "extra": "mean: 5.13482628418 msec\nrounds: 95"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cpu-32-2]",
            "value": 12025.9842503,
            "unit": "iter/sec",
            "range": "stddev: 0.00000250867749304",
            "extra": "mean: 83.1532770362 usec\nrounds: 4631"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cpu-32-8]",
            "value": 2992.49091548,
            "unit": "iter/sec",
            "range": "stddev: 0.00000470458795269",
            "extra": "mean: 334.169769682 usec\nrounds: 2405"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cpu-32-32]",
            "value": 755.458386598,
            "unit": "iter/sec",
            "range": "stddev: 0.00000988872454511",
            "extra": "mean: 1.32369964745 msec\nrounds: 692"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cpu-4096-2]",
            "value": 1477.66762657,
            "unit": "iter/sec",
            "range": "stddev: 0.00255020067031",
            "extra": "mean: 676.742172612 usec\nrounds: 29"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cpu-4096-8]",
            "value": 1142.9470697,
            "unit": "iter/sec",
            "range": "stddev: 0.0000638472971414",
            "extra": "mean: 874.931155177 usec\nrounds: 58"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cpu-4096-32]",
            "value": 226.231494863,
            "unit": "iter/sec",
            "range": "stddev: 0.0000590682538101",
            "extra": "mean: 4.42025103801 msec\nrounds: 105"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cpu-32-2]",
            "value": 16941.7848513,
            "unit": "iter/sec",
            "range": "stddev: 0.00000160569524168",
            "extra": "mean: 59.0256580859 usec\nrounds: 8362"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cpu-32-8]",
            "value": 4249.50398436,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033329664005",
            "extra": "mean: 235.321581926 usec\nrounds: 3631"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cpu-32-32]",
            "value": 1054.9244871,
            "unit": "iter/sec",
            "range": "stddev: 0.00000665858660949",
            "extra": "mean: 947.935148183 usec\nrounds: 985"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cpu-4096-2]",
            "value": 5969.90254822,
            "unit": "iter/sec",
            "range": "stddev: 0.0000382154172031",
            "extra": "mean: 167.506921917 usec\nrounds: 2676"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cpu-4096-8]",
            "value": 1395.09525617,
            "unit": "iter/sec",
            "range": "stddev: 0.0000228306131299",
            "extra": "mean: 716.796932377 usec\nrounds: 680"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cpu-4096-32]",
            "value": 264.163988562,
            "unit": "iter/sec",
            "range": "stddev: 0.0000599755222478",
            "extra": "mean: 3.78552733643 msec\nrounds: 113"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cpu-128-128]",
            "value": 217.219027803,
            "unit": "iter/sec",
            "range": "stddev: 0.0000358556771439",
            "extra": "mean: 4.60364826283 msec\nrounds: 175"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cpu-128-256]",
            "value": 97.2778988133,
            "unit": "iter/sec",
            "range": "stddev: 0.000122943589057",
            "extra": "mean: 10.279827301 msec\nrounds: 93"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cpu-4096-128]",
            "value": 15.2375861742,
            "unit": "iter/sec",
            "range": "stddev: 0.000227086279626",
            "extra": "mean: 65.6271924285 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cpu-4096-256]",
            "value": 6.86562762105,
            "unit": "iter/sec",
            "range": "stddev: 0.0016280387898",
            "extra": "mean: 145.653107799 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cpu-128-128]",
            "value": 34.7535728412,
            "unit": "iter/sec",
            "range": "stddev: 0.0000925727230601",
            "extra": "mean: 28.7740199999 msec\nrounds: 34"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cpu-128-256]",
            "value": 16.6030706717,
            "unit": "iter/sec",
            "range": "stddev: 0.000176167044995",
            "extra": "mean: 60.2298225293 msec\nrounds: 17"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cpu-4096-128]",
            "value": 8.26585022034,
            "unit": "iter/sec",
            "range": "stddev: 0.0011866240857",
            "extra": "mean: 120.979690334 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cpu-4096-256]",
            "value": 3.83221174982,
            "unit": "iter/sec",
            "range": "stddev: 0.00092466104315",
            "extra": "mean: 260.945914601 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cpu-128-128]",
            "value": 62.3592362442,
            "unit": "iter/sec",
            "range": "stddev: 0.000134174833496",
            "extra": "mean: 16.0361168646 msec\nrounds: 59"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cpu-128-256]",
            "value": 28.6415520644,
            "unit": "iter/sec",
            "range": "stddev: 0.000115860398545",
            "extra": "mean: 34.9143090344 msec\nrounds: 29"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cpu-4096-128]",
            "value": 9.75979318047,
            "unit": "iter/sec",
            "range": "stddev: 0.000483569099368",
            "extra": "mean: 102.4611876 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cpu-4096-256]",
            "value": 4.65819669371,
            "unit": "iter/sec",
            "range": "stddev: 0.000803182492837",
            "extra": "mean: 214.6753488 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cpu-128-128]",
            "value": 98.5019461024,
            "unit": "iter/sec",
            "range": "stddev: 0.0000316035737211",
            "extra": "mean: 10.1520836853 msec\nrounds: 89"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cpu-128-256]",
            "value": 44.4805911419,
            "unit": "iter/sec",
            "range": "stddev: 0.0000791420443309",
            "extra": "mean: 22.4817156051 msec\nrounds: 43"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cpu-4096-128]",
            "value": 10.3281136355,
            "unit": "iter/sec",
            "range": "stddev: 0.000583948162974",
            "extra": "mean: 96.8231020001 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cpu-4096-256]",
            "value": 4.7200928622,
            "unit": "iter/sec",
            "range": "stddev: 0.00218596856566",
            "extra": "mean: 211.8602386 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[None-batch_size0-4]",
            "value": 528303.255807,
            "unit": "iter/sec",
            "range": "stddev: 2.72656246022e-7",
            "extra": "mean: 1.89285223782 usec\nrounds: 54142"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[None-batch_size0-16]",
            "value": 327390.149353,
            "unit": "iter/sec",
            "range": "stddev: 3.77702249936e-7",
            "extra": "mean: 3.05445964693 usec\nrounds: 86431"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[None-batch_size0-64]",
            "value": 127672.215801,
            "unit": "iter/sec",
            "range": "stddev: 6.94952003258e-7",
            "extra": "mean: 7.83255772385 usec\nrounds: 51547"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[None-batch_size0-256]",
            "value": 37474.0986272,
            "unit": "iter/sec",
            "range": "stddev: 0.00000109980654032",
            "extra": "mean: 26.6850981513 usec\nrounds: 26296"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[None-batch_size1-4]",
            "value": 298944.434269,
            "unit": "iter/sec",
            "range": "stddev: 4.3599879511e-7",
            "extra": "mean: 3.34510325454 usec\nrounds: 53391"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[None-batch_size1-16]",
            "value": 118520.485969,
            "unit": "iter/sec",
            "range": "stddev: 7.1639471239e-7",
            "extra": "mean: 8.43735993678 usec\nrounds: 50098"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[None-batch_size1-64]",
            "value": 34333.4951941,
            "unit": "iter/sec",
            "range": "stddev: 0.00000127871632217",
            "extra": "mean: 29.1260762805 usec\nrounds: 25196"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[None-batch_size1-256]",
            "value": 9008.53038381,
            "unit": "iter/sec",
            "range": "stddev: 0.00000243008986414",
            "extra": "mean: 111.005897454 usec\nrounds: 8173"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[cpu-batch_size0-4]",
            "value": 319340.185177,
            "unit": "iter/sec",
            "range": "stddev: 4.4531046232e-7",
            "extra": "mean: 3.13145681758 usec\nrounds: 43669"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[cpu-batch_size0-16]",
            "value": 187630.020201,
            "unit": "iter/sec",
            "range": "stddev: 5.68321329369e-7",
            "extra": "mean: 5.32963754376 usec\nrounds: 51814"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[cpu-batch_size0-64]",
            "value": 70740.4501636,
            "unit": "iter/sec",
            "range": "stddev: 8.52402336058e-7",
            "extra": "mean: 14.1361837207 usec\nrounds: 34400"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[cpu-batch_size0-256]",
            "value": 20153.9984772,
            "unit": "iter/sec",
            "range": "stddev: 0.00000166811329255",
            "extra": "mean: 49.6179455968 usec\nrounds: 15366"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[cpu-batch_size1-4]",
            "value": 214949.256623,
            "unit": "iter/sec",
            "range": "stddev: 6.03978540808e-7",
            "extra": "mean: 4.65226079733 usec\nrounds: 48217"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[cpu-batch_size1-16]",
            "value": 91316.5719125,
            "unit": "iter/sec",
            "range": "stddev: 7.52307789264e-7",
            "extra": "mean: 10.9509148127 usec\nrounds: 42409"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[cpu-batch_size1-64]",
            "value": 27359.3189555,
            "unit": "iter/sec",
            "range": "stddev: 0.00000140549213974",
            "extra": "mean: 36.5506174194 usec\nrounds: 19713"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[cpu-batch_size1-256]",
            "value": 7259.03060499,
            "unit": "iter/sec",
            "range": "stddev: 0.00000284502429799",
            "extra": "mean: 137.759441228 usec\nrounds: 6586"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[constructor-dict-16]",
            "value": 38703.6980648,
            "unit": "iter/sec",
            "range": "stddev: 0.00000130329848747",
            "extra": "mean: 25.8373243385 usec\nrounds: 10603"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[constructor-dict-256]",
            "value": 2693.19358261,
            "unit": "iter/sec",
            "range": "stddev: 0.00000483414692546",
            "extra": "mean: 371.306394927 usec\nrounds: 2246"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[constructor-tuple_keys-16]",
            "value": 27500.5339979,
            "unit": "iter/sec",
            "range": "stddev: 0.00000136223625653",
            "extra": "mean: 36.3629302644 usec\nrounds: 14195"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[constructor-tuple_keys-256]",
            "value": 1852.28676053,
            "unit": "iter/sec",
            "range": "stddev: 0.00000607175106968",
            "extra": "mean: 539.873210405 usec\nrounds: 1711"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[constructor-ready_children-16]",
            "value": 138966.620183,
            "unit": "iter/sec",
            "range": "stddev: 6.21053365464e-7",
            "extra": "mean: 7.19597266369 usec\nrounds: 64351"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[constructor-ready_children-256]",
            "value": 13784.8552291,
            "unit": "iter/sec",
            "range": "stddev: 0.00000208724882499",
            "extra": "mean: 72.5433806437 usec\nrounds: 12760"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[set-dict-16]",
            "value": 35828.7581034,
            "unit": "iter/sec",
            "range": "stddev: 0.00000133375533227",
            "extra": "mean: 27.9105403853 usec\nrounds: 14414"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[set-dict-256]",
            "value": 2357.39589726,
            "unit": "iter/sec",
            "range": "stddev: 0.0000048423057202",
            "extra": "mean: 424.196886557 usec\nrounds: 1005"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[set-tuple_keys-16]",
            "value": 30000.7911219,
            "unit": "iter/sec",
            "range": "stddev: 0.00000131863789504",
            "extra": "mean: 33.3324543322 usec\nrounds: 16378"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[set-tuple_keys-256]",
            "value": 1977.44683622,
            "unit": "iter/sec",
            "range": "stddev: 0.00000572086498368",
            "extra": "mean: 505.702596745 usec\nrounds: 1835"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[set-ready_children-16]",
            "value": 112540.467885,
            "unit": "iter/sec",
            "range": "stddev: 6.30348219485e-7",
            "extra": "mean: 8.88569257615 usec\nrounds: 44623"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[set-ready_children-256]",
            "value": 8687.62073149,
            "unit": "iter/sec",
            "range": "stddev: 0.00000241639379279",
            "extra": "mean: 115.106314019 usec\nrounds: 7971"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[update-dict-16]",
            "value": 30897.6675273,
            "unit": "iter/sec",
            "range": "stddev: 0.00000138438031224",
            "extra": "mean: 32.3649025972 usec\nrounds: 11190"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[update-dict-256]",
            "value": 2121.12008311,
            "unit": "iter/sec",
            "range": "stddev: 0.00000577522662398",
            "extra": "mean: 471.449027316 usec\nrounds: 1063"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[update-tuple_keys-16]",
            "value": 23856.386785,
            "unit": "iter/sec",
            "range": "stddev: 0.00000153363930311",
            "extra": "mean: 41.9174960992 usec\nrounds: 12989"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[update-tuple_keys-256]",
            "value": 1583.13527007,
            "unit": "iter/sec",
            "range": "stddev: 0.00000601534706768",
            "extra": "mean: 631.657963097 usec\nrounds: 1490"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[update-ready_children-16]",
            "value": 79685.1054002,
            "unit": "iter/sec",
            "range": "stddev: 8.82913681434e-7",
            "extra": "mean: 12.5493967157 usec\nrounds: 33026"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[update-ready_children-256]",
            "value": 6501.5377585,
            "unit": "iter/sec",
            "range": "stddev: 0.00000294993648833",
            "extra": "mean: 153.809765804 usec\nrounds: 4573"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 117276.443541,
            "unit": "iter/sec",
            "range": "stddev: 6.92434862293e-7",
            "extra": "mean: 8.52686157431 usec\nrounds: 21282"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 117065.898438,
            "unit": "iter/sec",
            "range": "stddev: 7.20496548357e-7",
            "extra": "mean: 8.54219728671 usec\nrounds: 18744"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 114549.131204,
            "unit": "iter/sec",
            "range": "stddev: 6.8341576994e-7",
            "extra": "mean: 8.72987852018 usec\nrounds: 60133"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 115631.275567,
            "unit": "iter/sec",
            "range": "stddev: 7.07152483528e-7",
            "extra": "mean: 8.64817926722 usec\nrounds: 70672"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_td_batchfree",
            "value": 1341733.43932,
            "unit": "iter/sec",
            "range": "stddev: 1.81125052937e-7",
            "extra": "mean: 745.304522264 nsec\nrounds: 150830"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 287419.118391,
            "unit": "iter/sec",
            "range": "stddev: 4.40158505535e-7",
            "extra": "mean: 3.4792396748 usec\nrounds: 102250"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 4436.22464764,
            "unit": "iter/sec",
            "range": "stddev: 0.00000378164538596",
            "extra": "mean: 225.416898248 usec\nrounds: 3440"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 4417.24693128,
            "unit": "iter/sec",
            "range": "stddev: 0.00000382801037555",
            "extra": "mean: 226.385351681 usec\nrounds: 4126"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 18049.0101793,
            "unit": "iter/sec",
            "range": "stddev: 0.00000178552569867",
            "extra": "mean: 55.4047003167 usec\nrounds: 13648"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 4491.56965094,
            "unit": "iter/sec",
            "range": "stddev: 0.00000406919288869",
            "extra": "mean: 222.639317146 usec\nrounds: 3519"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 17763.5774905,
            "unit": "iter/sec",
            "range": "stddev: 0.00000173563257704",
            "extra": "mean: 56.2949665141 usec\nrounds: 13714"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 4395.19376614,
            "unit": "iter/sec",
            "range": "stddev: 0.00000377715683129",
            "extra": "mean: 227.521254627 usec\nrounds: 4143"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 864181.163617,
            "unit": "iter/sec",
            "range": "stddev: 2.83326833358e-7",
            "extra": "mean: 1.15716477297 usec\nrounds: 151286"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 38336.350896,
            "unit": "iter/sec",
            "range": "stddev: 0.00000116522390856",
            "extra": "mean: 26.0849031436 usec\nrounds: 12483"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 33210.2567608,
            "unit": "iter/sec",
            "range": "stddev: 0.0000047114965462",
            "extra": "mean: 30.1111794227 usec\nrounds: 25426"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 42062.811833,
            "unit": "iter/sec",
            "range": "stddev: 0.00000109206344016",
            "extra": "mean: 23.7739693668 usec\nrounds: 25439"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf_loop",
            "value": 70839.7522788,
            "unit": "iter/sec",
            "range": "stddev: 8.85501983388e-7",
            "extra": "mean: 14.1163678278 usec\nrounds: 42338"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf_small",
            "value": 233006.158709,
            "unit": "iter/sec",
            "range": "stddev: 4.79650271682e-7",
            "extra": "mean: 4.29173205352 usec\nrounds: 75586"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf_small_len",
            "value": 389120.845611,
            "unit": "iter/sec",
            "range": "stddev: 3.65926818395e-7",
            "extra": "mean: 2.56989573106 usec\nrounds: 107067"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 38991.2733903,
            "unit": "iter/sec",
            "range": "stddev: 0.00000123190399707",
            "extra": "mean: 25.6467643411 usec\nrounds: 22257"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 42137.636811,
            "unit": "iter/sec",
            "range": "stddev: 0.00000117992088772",
            "extra": "mean: 23.7317532657 usec\nrounds: 27301"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 33430.1663013,
            "unit": "iter/sec",
            "range": "stddev: 0.00000131680380326",
            "extra": "mean: 29.9131027644 usec\nrounds: 27071"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_lazy_stack_nested",
            "value": 4618.43760793,
            "unit": "iter/sec",
            "range": "stddev: 0.00000381387919297",
            "extra": "mean: 216.523440369 usec\nrounds: 3597"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_lazy_stack_nested_leaf",
            "value": 4512.05040866,
            "unit": "iter/sec",
            "range": "stddev: 0.00000366216817076",
            "extra": "mean: 221.628729608 usec\nrounds: 3643"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_lazy_stack_nested_leaf_loop",
            "value": 9025.43678905,
            "unit": "iter/sec",
            "range": "stddev: 0.00000264415581806",
            "extra": "mean: 110.797961736 usec\nrounds: 7554"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 1473951.60891,
            "unit": "iter/sec",
            "range": "stddev: 7.01459113416e-8",
            "extra": "mean: 678.448324869 nsec\nrounds: 199204"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 35397.9612925,
            "unit": "iter/sec",
            "range": "stddev: 0.00000124120968943",
            "extra": "mean: 28.2502145176 usec\nrounds: 19490"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 33144.7227435,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013766296495",
            "extra": "mean: 30.1707154934 usec\nrounds: 28337"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 32314.7307808,
            "unit": "iter/sec",
            "range": "stddev: 0.00000137828071755",
            "extra": "mean: 30.9456392129 usec\nrounds: 23964"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 35085.8814747,
            "unit": "iter/sec",
            "range": "stddev: 0.00000125986097297",
            "extra": "mean: 28.5014928504 usec\nrounds: 26738"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 32599.6188677,
            "unit": "iter/sec",
            "range": "stddev: 0.00000130977823803",
            "extra": "mean: 30.6752052548 usec\nrounds: 24808"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 33274.3273449,
            "unit": "iter/sec",
            "range": "stddev: 0.00000132491278729",
            "extra": "mean: 30.0531995623 usec\nrounds: 29395"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 1720056.27321,
            "unit": "iter/sec",
            "range": "stddev: 1.82039208194e-7",
            "extra": "mean: 581.376327959 nsec\nrounds: 129033"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 631325.454466,
            "unit": "iter/sec",
            "range": "stddev: 3.44846753649e-7",
            "extra": "mean: 1.58396908112 usec\nrounds: 69931"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 636685.748753,
            "unit": "iter/sec",
            "range": "stddev: 2.85901087878e-7",
            "extra": "mean: 1.57063355346 usec\nrounds: 141243"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 633944.783874,
            "unit": "iter/sec",
            "range": "stddev: 3.11561148328e-7",
            "extra": "mean: 1.57742444679 usec\nrounds: 135502"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 630261.629341,
            "unit": "iter/sec",
            "range": "stddev: 3.01874043099e-7",
            "extra": "mean: 1.58664267892 usec\nrounds: 150830"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 407200.208272,
            "unit": "iter/sec",
            "range": "stddev: 3.59397149484e-7",
            "extra": "mean: 2.45579442172 usec\nrounds: 71429"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 405290.177502,
            "unit": "iter/sec",
            "range": "stddev: 3.90116573452e-7",
            "extra": "mean: 2.46736796377 usec\nrounds: 116280"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 409160.854045,
            "unit": "iter/sec",
            "range": "stddev: 3.48540975043e-7",
            "extra": "mean: 2.4440265732 usec\nrounds: 95329"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 406766.484495,
            "unit": "iter/sec",
            "range": "stddev: 3.5163153615e-7",
            "extra": "mean: 2.45841296694 usec\nrounds: 106838"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 345393.724535,
            "unit": "iter/sec",
            "range": "stddev: 3.25270456428e-7",
            "extra": "mean: 2.89524658083 usec\nrounds: 86881"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 364246.011114,
            "unit": "iter/sec",
            "range": "stddev: 3.6423551734e-7",
            "extra": "mean: 2.74539725759 usec\nrounds: 149926"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 351182.867732,
            "unit": "iter/sec",
            "range": "stddev: 4.18247314856e-7",
            "extra": "mean: 2.84751931795 usec\nrounds: 132451"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 371902.328039,
            "unit": "iter/sec",
            "range": "stddev: 3.85448638667e-7",
            "extra": "mean: 2.68887803223 usec\nrounds: 142858"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 379954.550867,
            "unit": "iter/sec",
            "range": "stddev: 3.61155168847e-7",
            "extra": "mean: 2.63189372971 usec\nrounds: 99305"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 362117.662965,
            "unit": "iter/sec",
            "range": "stddev: 3.20977685783e-7",
            "extra": "mean: 2.76153334199 usec\nrounds: 121952"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 374492.02986,
            "unit": "iter/sec",
            "range": "stddev: 3.75462707651e-7",
            "extra": "mean: 2.67028379849 usec\nrounds: 99901"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 367002.812601,
            "unit": "iter/sec",
            "range": "stddev: 3.37309461738e-7",
            "extra": "mean: 2.7247747583 usec\nrounds: 135318"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_shallow_nested_getitemleaf",
            "value": 1175591.26647,
            "unit": "iter/sec",
            "range": "stddev: 8.18200568198e-8",
            "extra": "mean: 850.635785177 nsec\nrounds: 191571"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_is_non_tensor[tensor]",
            "value": 10308249.4959,
            "unit": "iter/sec",
            "range": "stddev: 6.3387031042e-9",
            "extra": "mean: 97.0096814597 nsec\nrounds: 97277"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_is_non_tensor[parameter]",
            "value": 3509094.09012,
            "unit": "iter/sec",
            "range": "stddev: 1.0584209501e-7",
            "extra": "mean: 284.973834932 nsec\nrounds: 141844"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_is_non_tensor[tensordict]",
            "value": 5488358.43076,
            "unit": "iter/sec",
            "range": "stddev: 9.06463920047e-9",
            "extra": "mean: 182.203843392 nsec\nrounds: 53362"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 4736.89865128,
            "unit": "iter/sec",
            "range": "stddev: 0.00000611425665234",
            "extra": "mean: 211.108591004 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 4380.25686111,
            "unit": "iter/sec",
            "range": "stddev: 0.0000149394825043",
            "extra": "mean: 228.297114007 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 4329.76627342,
            "unit": "iter/sec",
            "range": "stddev: 0.00000999539301536",
            "extra": "mean: 230.959349039 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 4037.85852529,
            "unit": "iter/sec",
            "range": "stddev: 0.0000203983685677",
            "extra": "mean: 247.65602701 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_data_nested_locked",
            "value": 17865.1225004,
            "unit": "iter/sec",
            "range": "stddev: 0.00000514610065473",
            "extra": "mean: 55.974987016 usec\nrounds: 10870"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 14486.3928542,
            "unit": "iter/sec",
            "range": "stddev: 0.00000218145482333",
            "extra": "mean: 69.0302969184 usec\nrounds: 8100"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 22372.195798,
            "unit": "iter/sec",
            "range": "stddev: 0.00000157070326716",
            "extra": "mean: 44.6983393596 usec\nrounds: 17159"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_wide_speed",
            "value": 46893.4067848,
            "unit": "iter/sec",
            "range": "stddev: 0.00000107350394342",
            "extra": "mean: 21.3249594893 usec\nrounds: 26296"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 3104.01988889,
            "unit": "iter/sec",
            "range": "stddev: 0.0000265468652267",
            "extra": "mean: 322.162884194 usec\nrounds: 1209"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 1007555.69832,
            "unit": "iter/sec",
            "range": "stddev: 0.00000173423057063",
            "extra": "mean: 992.50096215 nsec\nrounds: 112867"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_device[cpu]",
            "value": 860452.140062,
            "unit": "iter/sec",
            "range": "stddev: 2.43101412829e-7",
            "extra": "mean: 1.16217968838 usec\nrounds: 97362"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_device[device1]",
            "value": 770574.606038,
            "unit": "iter/sec",
            "range": "stddev: 2.23860175208e-7",
            "extra": "mean: 1.29773287643 usec\nrounds: 71634"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 404918.792533,
            "unit": "iter/sec",
            "range": "stddev: 3.95459797345e-7",
            "extra": "mean: 2.46963099377 usec\nrounds: 57671"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 191421.465588,
            "unit": "iter/sec",
            "range": "stddev: 4.80154726184e-7",
            "extra": "mean: 5.22407451498 usec\nrounds: 31776"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 164119.862608,
            "unit": "iter/sec",
            "range": "stddev: 5.00565374578e-7",
            "extra": "mean: 6.09310770864 usec\nrounds: 29104"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_many_keys[10]",
            "value": 162595.317825,
            "unit": "iter/sec",
            "range": "stddev: 5.02707937048e-7",
            "extra": "mean: 6.15023860081 usec\nrounds: 61350"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_many_keys[50]",
            "value": 41462.7203903,
            "unit": "iter/sec",
            "range": "stddev: 0.00000119622551262",
            "extra": "mean: 24.1180508801 usec\nrounds: 16096"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_many_keys[100]",
            "value": 21552.4315403,
            "unit": "iter/sec",
            "range": "stddev: 0.00000150134404212",
            "extra": "mean: 46.398477041 usec\nrounds: 17227"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_many_keys[10]",
            "value": 68433.6356926,
            "unit": "iter/sec",
            "range": "stddev: 7.94293024824e-7",
            "extra": "mean: 14.6126972487 usec\nrounds: 24261"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_many_keys[50]",
            "value": 19456.7394181,
            "unit": "iter/sec",
            "range": "stddev: 0.00000160757035906",
            "extra": "mean: 51.3960730271 usec\nrounds: 13171"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_from_namedtuple[namedtuple]",
            "value": 242686.413739,
            "unit": "iter/sec",
            "range": "stddev: 4.94076489363e-7",
            "extra": "mean: 4.1205438104 usec\nrounds: 18843"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_from_namedtuple[structseq]",
            "value": 225506.843626,
            "unit": "iter/sec",
            "range": "stddev: 4.07726417657e-7",
            "extra": "mean: 4.4344552206 usec\nrounds: 28843"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lazy_stack_flags",
            "value": 879085.920557,
            "unit": "iter/sec",
            "range": "stddev: 3.28277579978e-7",
            "extra": "mean: 1.1375452349 usec\nrounds: 76046"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 156130.678003,
            "unit": "iter/sec",
            "range": "stddev: 5.39553284338e-7",
            "extra": "mean: 6.4048911642 usec\nrounds: 16456"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone_device",
            "value": 57119.7553349,
            "unit": "iter/sec",
            "range": "stddev: 0.0000399776801495",
            "extra": "mean: 17.5070777901 usec\nrounds: 6298"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 116066.617587,
            "unit": "iter/sec",
            "range": "stddev: 0.0000273334459794",
            "extra": "mean: 8.61574172478 usec\nrounds: 14968"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice]",
            "value": 90505.0021974,
            "unit": "iter/sec",
            "range": "stddev: 0.00000620528675599",
            "extra": "mean: 11.0491130404 usec\nrounds: 15303"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 74395.5427873,
            "unit": "iter/sec",
            "range": "stddev: 0.00000593462088069",
            "extra": "mean: 13.4416654887 usec\nrounds: 14998"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 45888.1567412,
            "unit": "iter/sec",
            "range": "stddev: 0.00000676555054818",
            "extra": "mean: 21.7921152432 usec\nrounds: 6057"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 81664.425553,
            "unit": "iter/sec",
            "range": "stddev: 0.00000602205542768",
            "extra": "mean: 12.2452339954 usec\nrounds: 6479"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 49270.7691842,
            "unit": "iter/sec",
            "range": "stddev: 0.00000680274622429",
            "extra": "mean: 20.2960095115 usec\nrounds: 8407"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tensor]",
            "value": 62981.3413434,
            "unit": "iter/sec",
            "range": "stddev: 0.00000661072557191",
            "extra": "mean: 15.8777183634 usec\nrounds: 15509"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[mask]",
            "value": 40648.7227927,
            "unit": "iter/sec",
            "range": "stddev: 0.00000715454965404",
            "extra": "mean: 24.601018957 usec\nrounds: 5011"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[ellipsis]",
            "value": 65310.2154218,
            "unit": "iter/sec",
            "range": "stddev: 0.00000652495621253",
            "extra": "mean: 15.3115403699 usec\nrounds: 14342"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[none]",
            "value": 69262.1675286,
            "unit": "iter/sec",
            "range": "stddev: 0.00000673200269693",
            "extra": "mean: 14.4378964113 usec\nrounds: 14780"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[separated_tensors]",
            "value": 33636.2898944,
            "unit": "iter/sec",
            "range": "stddev: 0.00000700363298435",
            "extra": "mean: 29.7297949072 usec\nrounds: 4915"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_mask_many_leaves[None]",
            "value": 18759.760588,
            "unit": "iter/sec",
            "range": "stddev: 0.00000882593576025",
            "extra": "mean: 53.3055843281 usec\nrounds: 6161"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_mask_many_leaves[cpu]",
            "value": 14851.155114,
            "unit": "iter/sec",
            "range": "stddev: 0.00197914203929",
            "extra": "mean: 67.3348296699 usec\nrounds: 6916"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 86050.714478,
            "unit": "iter/sec",
            "range": "stddev: 0.00000106388017363",
            "extra": "mean: 11.6210540036 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 51729.5035398,
            "unit": "iter/sec",
            "range": "stddev: 0.00000119298874988",
            "extra": "mean: 19.331327996 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 39159.7277641,
            "unit": "iter/sec",
            "range": "stddev: 0.00000164269432138",
            "extra": "mean: 25.5364390177 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 63478.4730851,
            "unit": "iter/sec",
            "range": "stddev: 0.00000185149118477",
            "extra": "mean: 15.7533719922 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_non_tensor_rows",
            "value": 7363.52081371,
            "unit": "iter/sec",
            "range": "stddev: 0.0000104409359101",
            "extra": "mean: 135.804600177 usec\nrounds: 20"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cpu-None]",
            "value": 1768.12331645,
            "unit": "iter/sec",
            "range": "stddev: 0.00290997553583",
            "extra": "mean: 565.571411618 usec\nrounds: 51"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cpu-cpu]",
            "value": 21708.413441,
            "unit": "iter/sec",
            "range": "stddev: 0.00000717852339249",
            "extra": "mean: 46.0650891285 usec\nrounds: 11556"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_non_tensor_rows[tensor]",
            "value": 7022.0671643,
            "unit": "iter/sec",
            "range": "stddev: 0.00000310543873576",
            "extra": "mean: 142.408207812 usec\nrounds: 3662"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_non_tensor_rows[mask]",
            "value": 6647.83261051,
            "unit": "iter/sec",
            "range": "stddev: 0.00000338952940159",
            "extra": "mean: 150.424966841 usec\nrounds: 4071"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 128752.402758,
            "unit": "iter/sec",
            "range": "stddev: 7.25211589113e-7",
            "extra": "mean: 7.7668453449 usec\nrounds: 16981"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 130477.911452,
            "unit": "iter/sec",
            "range": "stddev: 6.05060840073e-7",
            "extra": "mean: 7.66413248702 usec\nrounds: 26476"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 8769.39803117,
            "unit": "iter/sec",
            "range": "stddev: 0.0000449738274793",
            "extra": "mean: 114.032912686 usec\nrounds: 3035"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 147325.764496,
            "unit": "iter/sec",
            "range": "stddev: 0.00000493371568429",
            "extra": "mean: 6.78767901475 usec\nrounds: 55804"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 92920.8245093,
            "unit": "iter/sec",
            "range": "stddev: 0.00000283782951137",
            "extra": "mean: 10.7618502664 usec\nrounds: 43123"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 90676.07738,
            "unit": "iter/sec",
            "range": "stddev: 8.4186363501e-7",
            "extra": "mean: 11.0282670898 usec\nrounds: 15591"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_many_leaves[False-False-16]",
            "value": 77593.5546594,
            "unit": "iter/sec",
            "range": "stddev: 8.02916477625e-7",
            "extra": "mean: 12.8876683687 usec\nrounds: 34759"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_many_leaves[False-False-256]",
            "value": 6006.32770932,
            "unit": "iter/sec",
            "range": "stddev: 0.00000312398074331",
            "extra": "mean: 166.491082138 usec\nrounds: 4676"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_many_leaves[False-False-1024]",
            "value": 1527.13770587,
            "unit": "iter/sec",
            "range": "stddev: 0.00000776816157739",
            "extra": "mean: 654.819795329 usec\nrounds: 1446"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_many_leaves[False-True-16]",
            "value": 23506.0420902,
            "unit": "iter/sec",
            "range": "stddev: 0.00000141650840138",
            "extra": "mean: 42.5422534412 usec\nrounds: 9075"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_many_leaves[False-True-256]",
            "value": 1907.51060901,
            "unit": "iter/sec",
            "range": "stddev: 0.00000623066933722",
            "extra": "mean: 524.243480103 usec\nrounds: 1410"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_many_leaves[False-True-1024]",
            "value": 481.880478804,
            "unit": "iter/sec",
            "range": "stddev: 0.0000558590807592",
            "extra": "mean: 2.07520338338 msec\nrounds: 446"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_many_leaves[True-False-16]",
            "value": 76076.9206633,
            "unit": "iter/sec",
            "range": "stddev: 8.39213130545e-7",
            "extra": "mean: 13.1445909125 usec\nrounds: 36232"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_many_leaves[True-False-256]",
            "value": 6007.01210016,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029766267411",
            "extra": "mean: 166.472113478 usec\nrounds: 5199"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_many_leaves[True-False-1024]",
            "value": 1525.16850504,
            "unit": "iter/sec",
            "range": "stddev: 0.00000698491934622",
            "extra": "mean: 655.665257114 usec\nrounds: 1439"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_many_leaves[True-True-16]",
            "value": 19537.1808561,
            "unit": "iter/sec",
            "range": "stddev: 0.00000161106554823",
            "extra": "mean: 51.1844573363 usec\nrounds: 7651"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_many_leaves[True-True-256]",
            "value": 1866.92242617,
            "unit": "iter/sec",
            "range": "stddev: 0.00000641848726984",
            "extra": "mean: 535.640895401 usec\nrounds: 1348"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_many_leaves[True-True-1024]",
            "value": 476.288197306,
            "unit": "iter/sec",
            "range": "stddev: 0.0000258317741592",
            "extra": "mean: 2.0995691383 msec\nrounds: 441"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__keys_to_update[tensors]",
            "value": 99747.7839983,
            "unit": "iter/sec",
            "range": "stddev: 8.02215691305e-7",
            "extra": "mean: 10.0252853739 usec\nrounds: 25854"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__keys_to_update[nested]",
            "value": 76259.9402676,
            "unit": "iter/sec",
            "range": "stddev: 8.34790198317e-7",
            "extra": "mean: 13.1130446272 usec\nrounds: 25228"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__keys_to_update[non-tensor]",
            "value": 54602.4636673,
            "unit": "iter/sec",
            "range": "stddev: 9.71745865818e-7",
            "extra": "mean: 18.31419194 usec\nrounds: 14447"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_copy__row_flat[64]",
            "value": 91406.3243515,
            "unit": "iter/sec",
            "range": "stddev: 0.00000614732256094",
            "extra": "mean: 10.9401620412 usec\nrounds: 11478"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_copy__row_flat[4096]",
            "value": 647.811268595,
            "unit": "iter/sec",
            "range": "stddev: 0.00522694213442",
            "extra": "mean: 1.54365947071 msec\nrounds: 51"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_tensor_leaves[False-dict-16]",
            "value": 118664.470461,
            "unit": "iter/sec",
            "range": "stddev: 6.35910670857e-7",
            "extra": "mean: 8.42712225582 usec\nrounds: 57837"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_tensor_leaves[False-dict-256]",
            "value": 8349.71768983,
            "unit": "iter/sec",
            "range": "stddev: 0.00000244853299707",
            "extra": "mean: 119.764528233 usec\nrounds: 6164"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_tensor_leaves[False-td-16]",
            "value": 110061.114295,
            "unit": "iter/sec",
            "range": "stddev: 6.73437631285e-7",
            "extra": "mean: 9.08586112732 usec\nrounds: 65147"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_tensor_leaves[False-td-256]",
            "value": 8282.40131267,
            "unit": "iter/sec",
            "range": "stddev: 0.00000261336878245",
            "extra": "mean: 120.737931217 usec\nrounds: 6179"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_tensor_leaves[False-kwargs-16]",
            "value": 97149.0400802,
            "unit": "iter/sec",
            "range": "stddev: 7.06431335969e-7",
            "extra": "mean: 10.2934624899 usec\nrounds: 55832"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_tensor_leaves[False-kwargs-256]",
            "value": 6806.41089534,
            "unit": "iter/sec",
            "range": "stddev: 0.00000293910670261",
            "extra": "mean: 146.920310186 usec\nrounds: 4855"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_tensor_leaves[True-dict-16]",
            "value": 82038.9077691,
            "unit": "iter/sec",
            "range": "stddev: 8.00665066324e-7",
            "extra": "mean: 12.1893382932 usec\nrounds: 45208"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_tensor_leaves[True-dict-256]",
            "value": 5548.67487507,
            "unit": "iter/sec",
            "range": "stddev: 0.00000321481653926",
            "extra": "mean: 180.223210499 usec\nrounds: 4119"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_tensor_leaves[True-td-16]",
            "value": 69365.9931899,
            "unit": "iter/sec",
            "range": "stddev: 8.17463691508e-7",
            "extra": "mean: 14.4162860505 usec\nrounds: 41824"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_tensor_leaves[True-td-256]",
            "value": 4809.10250642,
            "unit": "iter/sec",
            "range": "stddev: 0.00000350965822553",
            "extra": "mean: 207.939007053 usec\nrounds: 3695"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_tensor_leaves[True-kwargs-16]",
            "value": 77942.0666397,
            "unit": "iter/sec",
            "range": "stddev: 8.0661649663e-7",
            "extra": "mean: 12.8300421468 usec\nrounds: 44366"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_tensor_leaves[True-kwargs-256]",
            "value": 5302.14453249,
            "unit": "iter/sec",
            "range": "stddev: 0.00000325112911348",
            "extra": "mean: 188.602931111 usec\nrounds: 4050"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf[False-None-batch_size0]",
            "value": 1814172.30315,
            "unit": "iter/sec",
            "range": "stddev: 5.15919549367e-8",
            "extra": "mean: 551.215558889 nsec\nrounds: 162338"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf[False-None-batch_size1]",
            "value": 977407.832021,
            "unit": "iter/sec",
            "range": "stddev: 9.70405455319e-8",
            "extra": "mean: 1.02311437175 usec\nrounds: 191939"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf[False-cpu-batch_size0]",
            "value": 1496022.32257,
            "unit": "iter/sec",
            "range": "stddev: 5.63602399243e-8",
            "extra": "mean: 668.439223742 nsec\nrounds: 135136"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf[False-cpu-batch_size1]",
            "value": 891689.524455,
            "unit": "iter/sec",
            "range": "stddev: 1.01093502329e-7",
            "extra": "mean: 1.12146657842 usec\nrounds: 170649"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf[True-None-batch_size0]",
            "value": 891780.029459,
            "unit": "iter/sec",
            "range": "stddev: 1.09295464064e-7",
            "extra": "mean: 1.12135276298 usec\nrounds: 175747"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf[True-None-batch_size1]",
            "value": 634777.079421,
            "unit": "iter/sec",
            "range": "stddev: 1.58422279712e-7",
            "extra": "mean: 1.5753561879 usec\nrounds: 197629"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf[True-cpu-batch_size0]",
            "value": 802844.783673,
            "unit": "iter/sec",
            "range": "stddev: 1.19805148467e-7",
            "extra": "mean: 1.24557077574 usec\nrounds: 187970"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf[True-cpu-batch_size1]",
            "value": 591734.244789,
            "unit": "iter/sec",
            "range": "stddev: 1.59904430109e-7",
            "extra": "mean: 1.68994782507 usec\nrounds: 177620"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf_inplace[False-None-batch_size0]",
            "value": 555007.105345,
            "unit": "iter/sec",
            "range": "stddev: 3.02878136302e-7",
            "extra": "mean: 1.80177873467 usec\nrounds: 35002"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf_inplace[False-None-batch_size1]",
            "value": 438146.510583,
            "unit": "iter/sec",
            "range": "stddev: 3.35013122999e-7",
            "extra": "mean: 2.28234158175 usec\nrounds: 69980"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf_inplace[False-cpu-batch_size0]",
            "value": 521535.46918,
            "unit": "iter/sec",
            "range": "stddev: 3.05183480979e-7",
            "extra": "mean: 1.91741513108 usec\nrounds: 75931"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf_inplace[False-cpu-batch_size1]",
            "value": 416451.446229,
            "unit": "iter/sec",
            "range": "stddev: 3.05462823259e-7",
            "extra": "mean: 2.40124031038 usec\nrounds: 71175"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf_inplace[True-None-batch_size0]",
            "value": 415653.495819,
            "unit": "iter/sec",
            "range": "stddev: 3.53944738032e-7",
            "extra": "mean: 2.40585008922 usec\nrounds: 78493"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf_inplace[True-None-batch_size1]",
            "value": 348974.872677,
            "unit": "iter/sec",
            "range": "stddev: 3.3789784181e-7",
            "extra": "mean: 2.86553582591 usec\nrounds: 73638"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf_inplace[True-cpu-batch_size0]",
            "value": 393991.619232,
            "unit": "iter/sec",
            "range": "stddev: 3.04423577302e-7",
            "extra": "mean: 2.53812505441 usec\nrounds: 84603"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf_inplace[True-cpu-batch_size1]",
            "value": 335056.391386,
            "unit": "iter/sec",
            "range": "stddev: 4.27303488784e-7",
            "extra": "mean: 2.98457222637 usec\nrounds: 78309"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 117866.705324,
            "unit": "iter/sec",
            "range": "stddev: 6.75429768229e-7",
            "extra": "mean: 8.48416011337 usec\nrounds: 21566"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 88873.5798029,
            "unit": "iter/sec",
            "range": "stddev: 0.00000138973297354",
            "extra": "mean: 11.25193789 usec\nrounds: 22267"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unravel_key[str]",
            "value": 11878838.656,
            "unit": "iter/sec",
            "range": "stddev: 6.23101280568e-9",
            "extra": "mean: 84.1833136186 nsec\nrounds: 115208"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unravel_key[tuple]",
            "value": 3799380.42701,
            "unit": "iter/sec",
            "range": "stddev: 2.43442202268e-8",
            "extra": "mean: 263.20080845 nsec\nrounds: 175132"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unravel_key[nested_tuple]",
            "value": 1454135.12127,
            "unit": "iter/sec",
            "range": "stddev: 2.7089536905e-7",
            "extra": "mean: 687.694001315 nsec\nrounds: 122101"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unravel_key_list[str]",
            "value": 3564422.52128,
            "unit": "iter/sec",
            "range": "stddev: 2.61512725934e-8",
            "extra": "mean: 280.550353958 nsec\nrounds: 166390"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unravel_key_list[tuple]",
            "value": 1277954.95213,
            "unit": "iter/sec",
            "range": "stddev: 1.85126236642e-7",
            "extra": "mean: 782.500195592 nsec\nrounds: 151746"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unravel_key_list[mixed]",
            "value": 759451.582185,
            "unit": "iter/sec",
            "range": "stddev: 5.08276406881e-7",
            "extra": "mean: 1.31673963615 usec\nrounds: 140647"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unravel_key_to_tuple[str]",
            "value": 9295609.55903,
            "unit": "iter/sec",
            "range": "stddev: 7.18931100465e-9",
            "extra": "mean: 107.577668108 nsec\nrounds: 89687"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unravel_key_to_tuple[tuple]",
            "value": 4159870.93715,
            "unit": "iter/sec",
            "range": "stddev: 2.31886594176e-8",
            "extra": "mean: 240.392073483 nsec\nrounds: 199204"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unravel_key_to_tuple[nested_tuple]",
            "value": 1508860.92794,
            "unit": "iter/sec",
            "range": "stddev: 1.84188961904e-7",
            "extra": "mean: 662.751603863 nsec\nrounds: 173612"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_get_by_key[str]",
            "value": 1841256.69412,
            "unit": "iter/sec",
            "range": "stddev: 3.81952516909e-8",
            "extra": "mean: 543.107326205 nsec\nrounds: 88184"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_get_by_key[tuple]",
            "value": 1050395.58028,
            "unit": "iter/sec",
            "range": "stddev: 9.69801285533e-8",
            "extra": "mean: 952.022284533 nsec\nrounds: 198808"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_get_by_key[nested_tuple]",
            "value": 719022.769604,
            "unit": "iter/sec",
            "range": "stddev: 2.57810531557e-7",
            "extra": "mean: 1.39077654043 usec\nrounds: 178254"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_by_key[x]",
            "value": 5275453.00406,
            "unit": "iter/sec",
            "range": "stddev: 9.92001148462e-9",
            "extra": "mean: 189.55718101 nsec\nrounds: 51047"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_by_key[sub]",
            "value": 4855188.76168,
            "unit": "iter/sec",
            "range": "stddev: 2.02532073459e-8",
            "extra": "mean: 205.965215584 nsec\nrounds: 194932"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_by_key[text]",
            "value": 606719.173598,
            "unit": "iter/sec",
            "range": "stddev: 3.86969613811e-7",
            "extra": "mean: 1.64820899605 usec\nrounds: 57505"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_by_key[key3]",
            "value": 1095912.21644,
            "unit": "iter/sec",
            "range": "stddev: 1.85730939637e-7",
            "extra": "mean: 912.481843891 nsec\nrounds: 152672"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_get_nested_with_default[hit]",
            "value": 1001227.90845,
            "unit": "iter/sec",
            "range": "stddev: 1.99791178985e-7",
            "extra": "mean: 998.773597457 nsec\nrounds: 177620"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_get_nested_with_default[miss]",
            "value": 964424.369128,
            "unit": "iter/sec",
            "range": "stddev: 2.09838342197e-7",
            "extra": "mean: 1.03688794271 usec\nrounds: 138122"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 52430.3368225,
            "unit": "iter/sec",
            "range": "stddev: 0.00000139741363843",
            "extra": "mean: 19.0729272518 usec\nrounds: 9058"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 22520.7316224,
            "unit": "iter/sec",
            "range": "stddev: 0.00000162330441314",
            "extra": "mean: 44.4035307897 usec\nrounds: 15714"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 22438.6291099,
            "unit": "iter/sec",
            "range": "stddev: 0.00000199042183755",
            "extra": "mean: 44.5660024551 usec\nrounds: 13864"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_select_update_step[exclude]",
            "value": 168459.954305,
            "unit": "iter/sec",
            "range": "stddev: 5.38053661121e-7",
            "extra": "mean: 5.93612888074 usec\nrounds: 55097"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_select_update_step[select]",
            "value": 154535.090356,
            "unit": "iter/sec",
            "range": "stddev: 6.66507082284e-7",
            "extra": "mean: 6.47102219759 usec\nrounds: 46861"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_select_update_step[step]",
            "value": 60990.7912668,
            "unit": "iter/sec",
            "range": "stddev: 9.70259132837e-7",
            "extra": "mean: 16.3959177973 usec\nrounds: 23138"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 10303.0293663,
            "unit": "iter/sec",
            "range": "stddev: 0.00000474210873388",
            "extra": "mean: 97.0588323539 usec\nrounds: 173"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 1206423.70855,
            "unit": "iter/sec",
            "range": "stddev: 8.70709100159e-8",
            "extra": "mean: 828.896177115 nsec\nrounds: 194932"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_same_device",
            "value": 90721.7309699,
            "unit": "iter/sec",
            "range": "stddev: 7.21145812793e-7",
            "extra": "mean: 11.0227173722 usec\nrounds: 14255"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_dtype_nested",
            "value": 61713.7967132,
            "unit": "iter/sec",
            "range": "stddev: 9.08034105951e-7",
            "extra": "mean: 16.2038320969 usec\nrounds: 4729"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 4462.50554946,
            "unit": "iter/sec",
            "range": "stddev: 0.000751477378244",
            "extra": "mean: 224.089357182 usec\nrounds: 784"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 5140.9314521,
            "unit": "iter/sec",
            "range": "stddev: 0.00000354746390881",
            "extra": "mean: 194.517279469 usec\nrounds: 4498"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1664.50874882,
            "unit": "iter/sec",
            "range": "stddev: 0.00626846858928",
            "extra": "mean: 600.777857556 usec\nrounds: 1994"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stack_nested",
            "value": 3344.42928424,
            "unit": "iter/sec",
            "range": "stddev: 0.000017103385443",
            "extra": "mean: 299.004677633 usec\nrounds: 1914"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stack_rollout_steps",
            "value": 10626.981454,
            "unit": "iter/sec",
            "range": "stddev: 0.00000417147183896",
            "extra": "mean: 94.100098351 usec\nrounds: 4952"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_cat_nested",
            "value": 3684.75590997,
            "unit": "iter/sec",
            "range": "stddev: 0.00000501159111599",
            "extra": "mean: 271.38839707 usec\nrounds: 2201"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_pack_nested[pack]",
            "value": 38964.8381245,
            "unit": "iter/sec",
            "range": "stddev: 0.00000168315315701",
            "extra": "mean: 25.6641641062 usec\nrounds: 19616"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_pack_nested[unpack]",
            "value": 48325.8915283,
            "unit": "iter/sec",
            "range": "stddev: 0.00000112877430808",
            "extra": "mean: 20.6928412157 usec\nrounds: 9706"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_apply_nested",
            "value": 8907.15241956,
            "unit": "iter/sec",
            "range": "stddev: 0.00000269392469446",
            "extra": "mean: 112.269326143 usec\nrounds: 4728"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_pad_nested[True]",
            "value": 22667.5490303,
            "unit": "iter/sec",
            "range": "stddev: 0.00000172753398873",
            "extra": "mean: 44.1159297225 usec\nrounds: 4027"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_pad_nested[False]",
            "value": 24519.8859876,
            "unit": "iter/sec",
            "range": "stddev: 0.00000175164141824",
            "extra": "mean: 40.7832238905 usec\nrounds: 10242"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_pointwise_nested[neg]",
            "value": 80332.2025387,
            "unit": "iter/sec",
            "range": "stddev: 0.00000212962572948",
            "extra": "mean: 12.448308006 usec\nrounds: 8964"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_pointwise_nested[mul_scalar]",
            "value": 50184.4926936,
            "unit": "iter/sec",
            "range": "stddev: 0.00000136303285851",
            "extra": "mean: 19.9264742219 usec\nrounds: 4015"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_pointwise_nested[add]",
            "value": 47608.3624066,
            "unit": "iter/sec",
            "range": "stddev: 0.00000125390777049",
            "extra": "mean: 21.0047132363 usec\nrounds: 10263"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_shape_ops_nested[reshape]",
            "value": 50565.2422973,
            "unit": "iter/sec",
            "range": "stddev: 0.00000120415368672",
            "extra": "mean: 19.7764304998 usec\nrounds: 7619"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_shape_ops_nested[view_flatten]",
            "value": 22590.2882848,
            "unit": "iter/sec",
            "range": "stddev: 0.00000161743300139",
            "extra": "mean: 44.2668100288 usec\nrounds: 5422"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_shape_ops_nested[unsqueeze]",
            "value": 60717.566867,
            "unit": "iter/sec",
            "range": "stddev: 8.58506920883e-7",
            "extra": "mean: 16.4696981714 usec\nrounds: 12563"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_shape_ops_nested[unbind]",
            "value": 24337.5086202,
            "unit": "iter/sec",
            "range": "stddev: 0.00000320412233495",
            "extra": "mean: 41.0888400948 usec\nrounds: 8130"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_shape_ops_nested[clone]",
            "value": 129208.145772,
            "unit": "iter/sec",
            "range": "stddev: 7.46246070381e-7",
            "extra": "mean: 7.73945012541 usec\nrounds: 22909"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_shape_ops_nested[eq_all]",
            "value": 37680.7174474,
            "unit": "iter/sec",
            "range": "stddev: 0.00000142022860128",
            "extra": "mean: 26.5387728192 usec\nrounds: 6704"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 1392.86118358,
            "unit": "iter/sec",
            "range": "stddev: 0.0000220146657251",
            "extra": "mean: 717.946635163 usec\nrounds: 899"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 1426.84707226,
            "unit": "iter/sec",
            "range": "stddev: 0.00000833550895443",
            "extra": "mean: 700.845955705 usec\nrounds: 1151"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate_njt[False-None]",
            "value": 367.037018316,
            "unit": "iter/sec",
            "range": "stddev: 0.0000260873611142",
            "extra": "mean: 2.72452082514 msec\nrounds: 366"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 16154.5550821,
            "unit": "iter/sec",
            "range": "stddev: 0.0000266731189428",
            "extra": "mean: 61.9020452694 usec\nrounds: 3028"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 14524.2052924,
            "unit": "iter/sec",
            "range": "stddev: 0.0000280377981603",
            "extra": "mean: 68.8505828626 usec\nrounds: 5365"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 307914.388113,
            "unit": "iter/sec",
            "range": "stddev: 0.00000172249844212",
            "extra": "mean: 3.24765596738 usec\nrounds: 21911"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 4171113.28584,
            "unit": "iter/sec",
            "range": "stddev: 2.05230395981e-8",
            "extra": "mean: 239.744147777 nsec\nrounds: 110498"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 465308.327038,
            "unit": "iter/sec",
            "range": "stddev: 7.67875855373e-7",
            "extra": "mean: 2.14911262467 usec\nrounds: 35588"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 6511.78513496,
            "unit": "iter/sec",
            "range": "stddev: 0.0000231818331409",
            "extra": "mean: 153.567720567 usec\nrounds: 2333"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 5239.86324358,
            "unit": "iter/sec",
            "range": "stddev: 0.00000820379984494",
            "extra": "mean: 190.844675426 usec\nrounds: 3315"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 3427.44296293,
            "unit": "iter/sec",
            "range": "stddev: 0.000023561884486",
            "extra": "mean: 291.762696218 usec\nrounds: 2469"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 19.2127879822,
            "unit": "iter/sec",
            "range": "stddev: 0.000924126805484",
            "extra": "mean: 52.0486668007 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_archive",
            "value": 6.95794654266,
            "unit": "iter/sec",
            "range": "stddev: 0.000385270063019",
            "extra": "mean: 143.7205638 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_load_memmap",
            "value": 59.2259883892,
            "unit": "iter/sec",
            "range": "stddev: 0.000257098703061",
            "extra": "mean: 16.8844797224 msec\nrounds: 54"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_load_archive",
            "value": 95.460067168,
            "unit": "iter/sec",
            "range": "stddev: 0.0195843201316",
            "extra": "mean: 10.4755845001 msec\nrounds: 110"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_zarr",
            "value": 0.720045953851,
            "unit": "iter/sec",
            "range": "stddev: 0.0219551603294",
            "extra": "mean: 1.388800249 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_load_zarr",
            "value": 5.39095749848,
            "unit": "iter/sec",
            "range": "stddev: 0.00846260157618",
            "extra": "mean: 185.495804833 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 38.6442786349,
            "unit": "iter/sec",
            "range": "stddev: 0.00183399693965",
            "extra": "mean: 25.8770517998 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 19.2261870386,
            "unit": "iter/sec",
            "range": "stddev: 0.000458453251378",
            "extra": "mean: 52.0123932005 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 10.6781403769,
            "unit": "iter/sec",
            "range": "stddev: 0.088302548783",
            "extra": "mean: 93.6492652003 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 38.5201361813,
            "unit": "iter/sec",
            "range": "stddev: 0.00285540179486",
            "extra": "mean: 25.9604482002 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_filesystem",
            "value": 28.7245037495,
            "unit": "iter/sec",
            "range": "stddev: 0.000188144380657",
            "extra": "mean: 34.8134822005 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_filesystem",
            "value": 29.4490560456,
            "unit": "iter/sec",
            "range": "stddev: 0.00048365490989",
            "extra": "mean: 33.9569457999 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_consolidate[few_large-0]",
            "value": 382.75028366,
            "unit": "iter/sec",
            "range": "stddev: 0.00470189029671",
            "extra": "mean: 2.61266952029 msec\nrounds: 469"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_consolidate[few_large-8]",
            "value": 567.637281963,
            "unit": "iter/sec",
            "range": "stddev: 0.0018029981037",
            "extra": "mean: 1.7616883735 msec\nrounds: 83"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_consolidate[many_small-0]",
            "value": 15.438426957,
            "unit": "iter/sec",
            "range": "stddev: 0.00139872094602",
            "extra": "mean: 64.7734385626 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_consolidate[many_small-8]",
            "value": 18.9436456835,
            "unit": "iter/sec",
            "range": "stddev: 0.0418705064896",
            "extra": "mean: 52.7881494781 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_consolidate_to_file[few_large-0]",
            "value": 161.377247654,
            "unit": "iter/sec",
            "range": "stddev: 0.000131823498886",
            "extra": "mean: 6.19666040002 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_consolidate_to_file[few_large-8]",
            "value": 166.781479043,
            "unit": "iter/sec",
            "range": "stddev: 0.0000956148974031",
            "extra": "mean: 5.99586959979 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_consolidate_to_file[many_small-0]",
            "value": 37.775331548,
            "unit": "iter/sec",
            "range": "stddev: 0.00120197112029",
            "extra": "mean: 26.4723024 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_consolidate_to_file[many_small-8]",
            "value": 37.4419007472,
            "unit": "iter/sec",
            "range": "stddev: 0.00107707653445",
            "extra": "mean: 26.7080458001 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 59758.821533,
            "unit": "iter/sec",
            "range": "stddev: 0.0000067342243651",
            "extra": "mean: 16.733931064 usec\nrounds: 14101"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 38978.4060957,
            "unit": "iter/sec",
            "range": "stddev: 0.00000181858665294",
            "extra": "mean: 25.655230682 usec\nrounds: 8904"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 59911.3787999,
            "unit": "iter/sec",
            "range": "stddev: 0.0000066497358284",
            "extra": "mean: 16.6913200803 usec\nrounds: 16921"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 33819.4317838,
            "unit": "iter/sec",
            "range": "stddev: 0.00000130551636381",
            "extra": "mean: 29.5687995704 usec\nrounds: 9979"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 53003.6432745,
            "unit": "iter/sec",
            "range": "stddev: 0.00000697399649826",
            "extra": "mean: 18.8666276169 usec\nrounds: 12646"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 34710.5881166,
            "unit": "iter/sec",
            "range": "stddev: 0.00000152143523572",
            "extra": "mean: 28.8096530269 usec\nrounds: 11649"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 44362.7541429,
            "unit": "iter/sec",
            "range": "stddev: 0.00000693106470559",
            "extra": "mean: 22.5414318682 usec\nrounds: 11089"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 26246.6752324,
            "unit": "iter/sec",
            "range": "stddev: 0.00000151260059236",
            "extra": "mean: 38.100063766 usec\nrounds: 9565"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 46741.006552,
            "unit": "iter/sec",
            "range": "stddev: 0.00000698118739038",
            "extra": "mean: 21.3944900585 usec\nrounds: 9419"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 38393.9598012,
            "unit": "iter/sec",
            "range": "stddev: 0.00000117475276745",
            "extra": "mean: 26.0457635831 usec\nrounds: 10659"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-compile]",
            "value": 23330.0847406,
            "unit": "iter/sec",
            "range": "stddev: 0.00000796482888595",
            "extra": "mean: 42.863110491 usec\nrounds: 13503"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-eager]",
            "value": 14376.5933497,
            "unit": "iter/sec",
            "range": "stddev: 0.0000172916192979",
            "extra": "mean: 69.5575075179 usec\nrounds: 11688"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-compile]",
            "value": 48055.0254242,
            "unit": "iter/sec",
            "range": "stddev: 0.00000684218218062",
            "extra": "mean: 20.8094781175 usec\nrounds: 25847"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-eager]",
            "value": 14582.2165043,
            "unit": "iter/sec",
            "range": "stddev: 0.0000185762782373",
            "extra": "mean: 68.5766803492 usec\nrounds: 13055"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-compile]",
            "value": 41668.641831,
            "unit": "iter/sec",
            "range": "stddev: 0.00000164150676402",
            "extra": "mean: 23.9988623592 usec\nrounds: 26947"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-eager]",
            "value": 46258.5552005,
            "unit": "iter/sec",
            "range": "stddev: 0.00000108847670769",
            "extra": "mean: 21.6176228519 usec\nrounds: 40454"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-compile]",
            "value": 107787.809895,
            "unit": "iter/sec",
            "range": "stddev: 0.00000106609883103",
            "extra": "mean: 9.27748695308 usec\nrounds: 55804"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-eager]",
            "value": 25687.2728503,
            "unit": "iter/sec",
            "range": "stddev: 0.0000086369584868",
            "extra": "mean: 38.929784638 usec\nrounds: 21345"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-compile]",
            "value": 18278.0815528,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023463557252",
            "extra": "mean: 54.7103369197 usec\nrounds: 12240"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-eager]",
            "value": 11672.1567353,
            "unit": "iter/sec",
            "range": "stddev: 0.00000289039272024",
            "extra": "mean: 85.6739694883 usec\nrounds: 10324"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-compile]",
            "value": 38823.5978384,
            "unit": "iter/sec",
            "range": "stddev: 0.00000164545534946",
            "extra": "mean: 25.7575303598 usec\nrounds: 20077"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-eager]",
            "value": 34614.9162306,
            "unit": "iter/sec",
            "range": "stddev: 0.0000116397194385",
            "extra": "mean: 28.8892797931 usec\nrounds: 24361"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-compile]",
            "value": 21842.4081502,
            "unit": "iter/sec",
            "range": "stddev: 0.00000194358487955",
            "extra": "mean: 45.7824976588 usec\nrounds: 15177"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-eager]",
            "value": 8338.25172415,
            "unit": "iter/sec",
            "range": "stddev: 0.0000098428588092",
            "extra": "mean: 119.929216949 usec\nrounds: 7693"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-eager]",
            "value": 10947.37963,
            "unit": "iter/sec",
            "range": "stddev: 0.00000259560136262",
            "extra": "mean: 91.3460603175 usec\nrounds: 9747"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-compile]",
            "value": 18395.0215788,
            "unit": "iter/sec",
            "range": "stddev: 0.00000210465427099",
            "extra": "mean: 54.3625347606 usec\nrounds: 11912"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-eager]",
            "value": 36930.3783812,
            "unit": "iter/sec",
            "range": "stddev: 0.00000131860389087",
            "extra": "mean: 27.077978722 usec\nrounds: 24301"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-compile]",
            "value": 37862.5317072,
            "unit": "iter/sec",
            "range": "stddev: 0.00000167316067468",
            "extra": "mean: 26.4113347658 usec\nrounds: 19620"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-eager]",
            "value": 10609.4871972,
            "unit": "iter/sec",
            "range": "stddev: 0.00000947016323807",
            "extra": "mean: 94.2552624279 usec\nrounds: 9751"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-compile]",
            "value": 21863.4580093,
            "unit": "iter/sec",
            "range": "stddev: 0.00000191716210726",
            "extra": "mean: 45.7384188529 usec\nrounds: 15514"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-compile]",
            "value": 64661.7845649,
            "unit": "iter/sec",
            "range": "stddev: 9.85852818036e-7",
            "extra": "mean: 15.4650850843 usec\nrounds: 34177"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-eager]",
            "value": 89776.4699224,
            "unit": "iter/sec",
            "range": "stddev: 6.57394890261e-7",
            "extra": "mean: 11.1387761277 usec\nrounds: 71943"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-compile]",
            "value": 82137.1796964,
            "unit": "iter/sec",
            "range": "stddev: 0.00000353619524442",
            "extra": "mean: 12.1747545228 usec\nrounds: 46729"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-eager]",
            "value": 30987.3077233,
            "unit": "iter/sec",
            "range": "stddev: 0.00000859226359587",
            "extra": "mean: 32.2712772897 usec\nrounds: 26330"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-compile]",
            "value": 6803.99445741,
            "unit": "iter/sec",
            "range": "stddev: 0.00000494698063672",
            "extra": "mean: 146.972488919 usec\nrounds: 3788"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-eager]",
            "value": 1120.04685724,
            "unit": "iter/sec",
            "range": "stddev: 0.0000125073640988",
            "extra": "mean: 892.819790119 usec\nrounds: 1091"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-compile]",
            "value": 8015.57198049,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033925955927",
            "extra": "mean: 124.757160491 usec\nrounds: 5284"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-eager]",
            "value": 2271.62018136,
            "unit": "iter/sec",
            "range": "stddev: 0.0000163914124184",
            "extra": "mean: 440.214437345 usec\nrounds: 2186"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add_stack[compile]",
            "value": 3058.43253291,
            "unit": "iter/sec",
            "range": "stddev: 0.00000787041384502",
            "extra": "mean: 326.964871463 usec\nrounds: 1968"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add_stack[eager]",
            "value": 807.736406121,
            "unit": "iter/sec",
            "range": "stddev: 0.0000167237034663",
            "extra": "mean: 1.23802764419 msec\nrounds: 787"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-compile]",
            "value": 43469.6009958,
            "unit": "iter/sec",
            "range": "stddev: 0.00000185024799718",
            "extra": "mean: 23.0045819858 usec\nrounds: 23117"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-eager]",
            "value": 63602.7974666,
            "unit": "iter/sec",
            "range": "stddev: 0.00000749197172991",
            "extra": "mean: 15.7225788774 usec\nrounds: 43123"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-compile]",
            "value": 65689.8358292,
            "unit": "iter/sec",
            "range": "stddev: 0.00000154122731883",
            "extra": "mean: 15.2230552471 usec\nrounds: 34953"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-eager]",
            "value": 70530.9734159,
            "unit": "iter/sec",
            "range": "stddev: 0.00000834173693368",
            "extra": "mean: 14.1781681376 usec\nrounds: 50685"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-compile]",
            "value": 62880.0054382,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013384976654",
            "extra": "mean: 15.9033065126 usec\nrounds: 41238"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-eager]",
            "value": 70947.8476009,
            "unit": "iter/sec",
            "range": "stddev: 0.00000810091851335",
            "extra": "mean: 14.0948602927 usec\nrounds: 51627"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-compile]",
            "value": 34713.9361961,
            "unit": "iter/sec",
            "range": "stddev: 0.00000396658355963",
            "extra": "mean: 28.806874402 usec\nrounds: 18359"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-eager]",
            "value": 86837.9301468,
            "unit": "iter/sec",
            "range": "stddev: 0.00000736868668717",
            "extra": "mean: 11.5157051568 usec\nrounds: 62657"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-compile]",
            "value": 46287.8526798,
            "unit": "iter/sec",
            "range": "stddev: 0.00000314757282812",
            "extra": "mean: 21.6039401723 usec\nrounds: 24704"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-eager]",
            "value": 86027.9189658,
            "unit": "iter/sec",
            "range": "stddev: 0.00000818581794321",
            "extra": "mean: 11.6241333281 usec\nrounds: 63212"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-compile]",
            "value": 45338.4540797,
            "unit": "iter/sec",
            "range": "stddev: 0.00000156591189598",
            "extra": "mean: 22.0563320982 usec\nrounds: 27353"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-eager]",
            "value": 85739.6308663,
            "unit": "iter/sec",
            "range": "stddev: 0.00000822373750349",
            "extra": "mean: 11.6632179296 usec\nrounds: 60828"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-compile]",
            "value": 34041.9384289,
            "unit": "iter/sec",
            "range": "stddev: 0.00000205660924706",
            "extra": "mean: 29.3755304824 usec\nrounds: 21111"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-eager]",
            "value": 86788.8171214,
            "unit": "iter/sec",
            "range": "stddev: 0.00000752800664977",
            "extra": "mean: 11.5222217927 usec\nrounds: 61426"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-compile]",
            "value": 45110.1412528,
            "unit": "iter/sec",
            "range": "stddev: 0.00000159483326798",
            "extra": "mean: 22.1679642809 usec\nrounds: 27518"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-eager]",
            "value": 86376.9966316,
            "unit": "iter/sec",
            "range": "stddev: 0.0000078861266178",
            "extra": "mean: 11.5771564073 usec\nrounds: 62306"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-compile]",
            "value": 45431.6628291,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016175704309",
            "extra": "mean: 22.011080769 usec\nrounds: 28401"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-eager]",
            "value": 86950.572071,
            "unit": "iter/sec",
            "range": "stddev: 0.00000789519306049",
            "extra": "mean: 11.5007868975 usec\nrounds: 63533"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-eager]",
            "value": 84153.912014,
            "unit": "iter/sec",
            "range": "stddev: 0.00000106897628153",
            "extra": "mean: 11.882988872 usec\nrounds: 44564"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-compile]",
            "value": 51237.1725022,
            "unit": "iter/sec",
            "range": "stddev: 0.00000162451007662",
            "extra": "mean: 19.5170801035 usec\nrounds: 25806"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-eager]",
            "value": 7760.09130347,
            "unit": "iter/sec",
            "range": "stddev: 0.00000434045949196",
            "extra": "mean: 128.864463174 usec\nrounds: 6734"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-compile]",
            "value": 46013.5485146,
            "unit": "iter/sec",
            "range": "stddev: 0.00000183761322002",
            "extra": "mean: 21.7327294304 usec\nrounds: 23502"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[eager]",
            "value": 33646.7933975,
            "unit": "iter/sec",
            "range": "stddev: 0.00000159986158246",
            "extra": "mean: 29.7205141716 usec\nrounds: 30249"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[compile]",
            "value": 51525.7992103,
            "unit": "iter/sec",
            "range": "stddev: 0.00000157755320783",
            "extra": "mean: 19.4077533066 usec\nrounds: 28225"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[eager]",
            "value": 28620.1283676,
            "unit": "iter/sec",
            "range": "stddev: 0.00000190661401315",
            "extra": "mean: 34.9404442621 usec\nrounds: 24814"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[compile]",
            "value": 38758.6362126,
            "unit": "iter/sec",
            "range": "stddev: 0.0000117794333842",
            "extra": "mean: 25.8007014105 usec\nrounds: 20912"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-eager]",
            "value": 177639.786009,
            "unit": "iter/sec",
            "range": "stddev: 6.64583219862e-7",
            "extra": "mean: 5.62936953746 usec\nrounds: 120337"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-compile]",
            "value": 79520.3003516,
            "unit": "iter/sec",
            "range": "stddev: 0.00000119475516832",
            "extra": "mean: 12.5754052183 usec\nrounds: 42000"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-eager]",
            "value": 107333.773313,
            "unit": "iter/sec",
            "range": "stddev: 8.22880181983e-7",
            "extra": "mean: 9.31673199528 usec\nrounds: 73965"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-compile]",
            "value": 65918.6406433,
            "unit": "iter/sec",
            "range": "stddev: 0.00000184661001292",
            "extra": "mean: 15.1702157423 usec\nrounds: 33591"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-eager]",
            "value": 62455.1575174,
            "unit": "iter/sec",
            "range": "stddev: 0.00000108143474652",
            "extra": "mean: 16.0114879179 usec\nrounds: 51976"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-compile]",
            "value": 48320.764167,
            "unit": "iter/sec",
            "range": "stddev: 0.00000226566223441",
            "extra": "mean: 20.6950369523 usec\nrounds: 27390"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[eager]",
            "value": 64839.2299524,
            "unit": "iter/sec",
            "range": "stddev: 0.00000117523195191",
            "extra": "mean: 15.4227618176 usec\nrounds: 53880"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[compile]",
            "value": 42487.7055679,
            "unit": "iter/sec",
            "range": "stddev: 0.00000142483865695",
            "extra": "mean: 23.5362203403 usec\nrounds: 23409"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_build_and_set_cold",
            "value": 4.60415759878,
            "unit": "iter/sec",
            "range": "stddev: 0.127378511191",
            "extra": "mean: 217.194997901 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[eager]",
            "value": 87817.1399318,
            "unit": "iter/sec",
            "range": "stddev: 9.50951271983e-7",
            "extra": "mean: 11.3872986615 usec\nrounds: 47529"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile]",
            "value": 36066.9233556,
            "unit": "iter/sec",
            "range": "stddev: 0.00000667328270722",
            "extra": "mean: 27.7262352029 usec\nrounds: 8036"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile-overhead]",
            "value": 35671.622927,
            "unit": "iter/sec",
            "range": "stddev: 0.0000187690913737",
            "extra": "mean: 28.0334876281 usec\nrounds: 7274"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[eager]",
            "value": 4226.21036415,
            "unit": "iter/sec",
            "range": "stddev: 0.0000438067697291",
            "extra": "mean: 236.61860481 usec\nrounds: 3854"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile]",
            "value": 3602.95686114,
            "unit": "iter/sec",
            "range": "stddev: 0.0000158307491727",
            "extra": "mean: 277.549812152 usec\nrounds: 2151"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile-overhead]",
            "value": 3722.03886969,
            "unit": "iter/sec",
            "range": "stddev: 0.0000135087875342",
            "extra": "mean: 268.669950801 usec\nrounds: 2215"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[eager]",
            "value": 184.979700065,
            "unit": "iter/sec",
            "range": "stddev: 0.000213157683436",
            "extra": "mean: 5.40599860228 msec\nrounds: 181"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile]",
            "value": 172.321643617,
            "unit": "iter/sec",
            "range": "stddev: 0.000280654476016",
            "extra": "mean: 5.8031015664 msec\nrounds: 173"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile-overhead]",
            "value": 177.507458272,
            "unit": "iter/sec",
            "range": "stddev: 0.000216379923044",
            "extra": "mean: 5.63356610328 msec\nrounds: 155"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[eager]",
            "value": 27947.1199498,
            "unit": "iter/sec",
            "range": "stddev: 0.00000170170496552",
            "extra": "mean: 35.7818623815 usec\nrounds: 18951"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile]",
            "value": 28565.3368683,
            "unit": "iter/sec",
            "range": "stddev: 0.00000192212000559",
            "extra": "mean: 35.0074639278 usec\nrounds: 13720"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile-overhead]",
            "value": 28909.2612352,
            "unit": "iter/sec",
            "range": "stddev: 0.00000258128121846",
            "extra": "mean: 34.5909911659 usec\nrounds: 15056"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[eager]",
            "value": 3627.46291013,
            "unit": "iter/sec",
            "range": "stddev: 0.0000261303987815",
            "extra": "mean: 275.674769053 usec\nrounds: 3165"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile]",
            "value": 3407.3669475,
            "unit": "iter/sec",
            "range": "stddev: 0.0000268819738601",
            "extra": "mean: 293.481745702 usec\nrounds: 2859"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile-overhead]",
            "value": 3474.40901314,
            "unit": "iter/sec",
            "range": "stddev: 0.0000214762360619",
            "extra": "mean: 287.818732975 usec\nrounds: 2865"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-eager]",
            "value": 4591.34303641,
            "unit": "iter/sec",
            "range": "stddev: 0.0000135514987088",
            "extra": "mean: 217.801195003 usec\nrounds: 4174"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile]",
            "value": 3999.07572509,
            "unit": "iter/sec",
            "range": "stddev: 0.00000836309769863",
            "extra": "mean: 250.057780532 usec\nrounds: 3135"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile-overhead]",
            "value": 4089.88079525,
            "unit": "iter/sec",
            "range": "stddev: 0.00000797879653247",
            "extra": "mean: 244.505903733 usec\nrounds: 3293"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-eager]",
            "value": 2045.29358249,
            "unit": "iter/sec",
            "range": "stddev: 0.00000872032929614",
            "extra": "mean: 488.927364051 usec\nrounds: 1769"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile]",
            "value": 3733.93567824,
            "unit": "iter/sec",
            "range": "stddev: 0.00000694044804414",
            "extra": "mean: 267.813933119 usec\nrounds: 2826"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile-overhead]",
            "value": 3767.390887,
            "unit": "iter/sec",
            "range": "stddev: 0.00000878018059218",
            "extra": "mean: 265.435690109 usec\nrounds: 3056"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-eager]",
            "value": 4624.77710171,
            "unit": "iter/sec",
            "range": "stddev: 0.000014376166105",
            "extra": "mean: 216.226637092 usec\nrounds: 4122"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile]",
            "value": 4043.80391276,
            "unit": "iter/sec",
            "range": "stddev: 0.00000780284016171",
            "extra": "mean: 247.291911668 usec\nrounds: 3271"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile-overhead]",
            "value": 4184.53299038,
            "unit": "iter/sec",
            "range": "stddev: 0.00000660122325923",
            "extra": "mean: 238.975293611 usec\nrounds: 3147"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-eager]",
            "value": 1929.81283783,
            "unit": "iter/sec",
            "range": "stddev: 0.00000921927643607",
            "extra": "mean: 518.184966126 usec\nrounds: 1773"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile]",
            "value": 3069.94517441,
            "unit": "iter/sec",
            "range": "stddev: 0.00000771519833386",
            "extra": "mean: 325.738716228 usec\nrounds: 2495"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile-overhead]",
            "value": 3091.83492043,
            "unit": "iter/sec",
            "range": "stddev: 0.00000826554437137",
            "extra": "mean: 323.432533022 usec\nrounds: 2497"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[eager]",
            "value": 587.582617808,
            "unit": "iter/sec",
            "range": "stddev: 0.0000496897271408",
            "extra": "mean: 1.70188832973 msec\nrounds: 549"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile]",
            "value": 1179.49957451,
            "unit": "iter/sec",
            "range": "stddev: 0.0000861699047251",
            "extra": "mean: 847.817177396 usec\nrounds: 885"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile-overhead]",
            "value": 1206.98695647,
            "unit": "iter/sec",
            "range": "stddev: 0.0000481758981372",
            "extra": "mean: 828.509367593 usec\nrounds: 974"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 168122.665846,
            "unit": "iter/sec",
            "range": "stddev: 3.30821740268e-7",
            "extra": "mean: 5.94803796957 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 88240.5212171,
            "unit": "iter/sec",
            "range": "stddev: 6.48459256308e-7",
            "extra": "mean: 11.3326619812 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 113960.113697,
            "unit": "iter/sec",
            "range": "stddev: 6.49884874044e-7",
            "extra": "mean: 8.77500002025 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 65782.3631372,
            "unit": "iter/sec",
            "range": "stddev: 0.00000136660274496",
            "extra": "mean: 15.2016429984 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_tdmodules",
            "value": 12471.9233579,
            "unit": "iter/sec",
            "range": "stddev: 0.00000354698707586",
            "extra": "mean: 80.1800950267 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_selected_out_keys",
            "value": 16972.5916896,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024560160885",
            "extra": "mean: 58.9185210065 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_probseq",
            "value": 17161.6277472,
            "unit": "iter/sec",
            "range": "stddev: 0.00000296707387848",
            "extra": "mean: 58.2695309986 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 646.729146691,
            "unit": "iter/sec",
            "range": "stddev: 0.0000166719963535",
            "extra": "mean: 1.54624235682 msec\nrounds: 454"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 9870.63816962,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029637202512",
            "extra": "mean: 101.310572104 usec\nrounds: 4090"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 12559.6352342,
            "unit": "iter/sec",
            "range": "stddev: 0.00000265114769338",
            "extra": "mean: 79.6201467124 usec\nrounds: 5739"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 5887.91516953,
            "unit": "iter/sec",
            "range": "stddev: 0.00000417740016673",
            "extra": "mean: 169.839403457 usec\nrounds: 3289"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_soft_update_lerp",
            "value": 7190.65848943,
            "unit": "iter/sec",
            "range": "stddev: 0.00000548502102853",
            "extra": "mean: 139.069321881 usec\nrounds: 3085"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1708.66455239,
            "unit": "iter/sec",
            "range": "stddev: 0.0000560222810367",
            "extra": "mean: 585.252382394 usec\nrounds: 1611"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1705.04475845,
            "unit": "iter/sec",
            "range": "stddev: 0.0000553394799756",
            "extra": "mean: 586.494867682 usec\nrounds: 1625"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1924.42686523,
            "unit": "iter/sec",
            "range": "stddev: 0.0000507660452925",
            "extra": "mean: 519.635231698 usec\nrounds: 1843"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1988.92410467,
            "unit": "iter/sec",
            "range": "stddev: 0.0000317537704239",
            "extra": "mean: 502.784393659 usec\nrounds: 1834"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 620.058481133,
            "unit": "iter/sec",
            "range": "stddev: 0.0000121183037073",
            "extra": "mean: 1.61275110401 msec\nrounds: 538"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 617.885363294,
            "unit": "iter/sec",
            "range": "stddev: 0.0000143570749568",
            "extra": "mean: 1.61842318884 msec\nrounds: 503"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_from_module_eager[4-False]",
            "value": 37443.1845831,
            "unit": "iter/sec",
            "range": "stddev: 0.00000121823045992",
            "extra": "mean: 26.7071300461 usec\nrounds: 7482"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_from_module_eager[4-True]",
            "value": 5744.21971677,
            "unit": "iter/sec",
            "range": "stddev: 0.00000371730065137",
            "extra": "mean: 174.088048387 usec\nrounds: 2811"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_from_module_eager[64-False]",
            "value": 2503.12160655,
            "unit": "iter/sec",
            "range": "stddev: 0.00000498046197109",
            "extra": "mean: 399.501165817 usec\nrounds: 1737"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_from_module_eager[64-True]",
            "value": 410.446420449,
            "unit": "iter/sec",
            "range": "stddev: 0.00002628641464",
            "extra": "mean: 2.43637159487 msec\nrounds: 353"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_from_module_eager[transformer-False]",
            "value": 9860.93037132,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022572082451",
            "extra": "mean: 101.410309407 usec\nrounds: 6939"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_from_module_eager[transformer-True]",
            "value": 4157.1867286,
            "unit": "iter/sec",
            "range": "stddev: 0.00000387636473867",
            "extra": "mean: 240.547289618 usec\nrounds: 2690"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_eager[4-no_swap]",
            "value": 13028.4326795,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019681459943",
            "extra": "mean: 76.755203377 usec\nrounds: 7675"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_eager[4-swap]",
            "value": 11507.3826442,
            "unit": "iter/sec",
            "range": "stddev: 0.00000199478164802",
            "extra": "mean: 86.9007341568 usec\nrounds: 6884"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_eager[4-context]",
            "value": 5830.23201199,
            "unit": "iter/sec",
            "range": "stddev: 0.00000310494628076",
            "extra": "mean: 171.519760782 usec\nrounds: 4368"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_eager[64-no_swap]",
            "value": 851.260458419,
            "unit": "iter/sec",
            "range": "stddev: 0.0000100950951927",
            "extra": "mean: 1.1747285923 msec\nrounds: 645"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_eager[64-swap]",
            "value": 722.62707064,
            "unit": "iter/sec",
            "range": "stddev: 0.0000125955930927",
            "extra": "mean: 1.38383966036 msec\nrounds: 583"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_eager[64-context]",
            "value": 373.069192817,
            "unit": "iter/sec",
            "range": "stddev: 0.0000225000763694",
            "extra": "mean: 2.68046791119 msec\nrounds: 349"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_eager[transformer-no_swap]",
            "value": 3830.97771535,
            "unit": "iter/sec",
            "range": "stddev: 0.00000384982618285",
            "extra": "mean: 261.029970494 usec\nrounds: 2950"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_eager[transformer-swap]",
            "value": 3400.99849133,
            "unit": "iter/sec",
            "range": "stddev: 0.00000419388399952",
            "extra": "mean: 294.031297735 usec\nrounds: 2603"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_eager[transformer-context]",
            "value": 1747.45458725,
            "unit": "iter/sec",
            "range": "stddev: 0.00000666173979632",
            "extra": "mean: 572.260937306 usec\nrounds: 1515"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_context_small[False-True]",
            "value": 16260.3405969,
            "unit": "iter/sec",
            "range": "stddev: 0.00000177257374389",
            "extra": "mean: 61.4993267848 usec\nrounds: 8495"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_context_small[False-False]",
            "value": 36234.0829354,
            "unit": "iter/sec",
            "range": "stddev: 0.00000112626805553",
            "extra": "mean: 27.5983250848 usec\nrounds: 16088"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_context_small[True-True]",
            "value": 15347.3142036,
            "unit": "iter/sec",
            "range": "stddev: 0.00000191831860784",
            "extra": "mean: 65.1579805256 usec\nrounds: 9349"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_context_small[True-False]",
            "value": 31468.6753862,
            "unit": "iter/sec",
            "range": "stddev: 0.00000128937983812",
            "extra": "mean: 31.7776324465 usec\nrounds: 15886"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_inplace_copy[False-small]",
            "value": 9854.4212037,
            "unit": "iter/sec",
            "range": "stddev: 0.00000235719970996",
            "extra": "mean: 101.477294235 usec\nrounds: 7290"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_inplace_copy[False-many_leaves]",
            "value": 631.394450604,
            "unit": "iter/sec",
            "range": "stddev: 0.00000950166072923",
            "extra": "mean: 1.58379599162 msec\nrounds: 597"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_inplace_copy[False-16MiB]",
            "value": 1061.5849804,
            "unit": "iter/sec",
            "range": "stddev: 0.0000770560896087",
            "extra": "mean: 941.987705611 usec\nrounds: 771"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_inplace_copy[True-small]",
            "value": 6074.67083393,
            "unit": "iter/sec",
            "range": "stddev: 0.00000345160446489",
            "extra": "mean: 164.617973111 usec\nrounds: 3978"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_inplace_copy[True-many_leaves]",
            "value": 371.598636177,
            "unit": "iter/sec",
            "range": "stddev: 0.0000184830475713",
            "extra": "mean: 2.69107553862 msec\nrounds: 349"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_inplace_copy[True-16MiB]",
            "value": 519.995325119,
            "unit": "iter/sec",
            "range": "stddev: 0.0000994773526651",
            "extra": "mean: 1.92309421199 msec\nrounds: 401"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_plain_tensor_speed[None]",
            "value": 613.221977039,
            "unit": "iter/sec",
            "range": "stddev: 0.0000167063896188",
            "extra": "mean: 1.63073085676 msec\nrounds: 342"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_plain_tensor_speed[False]",
            "value": 1314.79867087,
            "unit": "iter/sec",
            "range": "stddev: 0.00000852460748438",
            "extra": "mean: 760.572718968 usec\nrounds: 975"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_plain_tensor_speed[True]",
            "value": 609.151735736,
            "unit": "iter/sec",
            "range": "stddev: 0.0000143978798953",
            "extra": "mean: 1.64162710428 msec\nrounds: 345"
          },
          {
            "name": "benchmarks/storage/test_h5_benchmarks.py::test_h5_index_many_keys[slice]",
            "value": 7.56097007965,
            "unit": "iter/sec",
            "range": "stddev: 0.000243316190617",
            "extra": "mean: 132.25816125 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/storage/test_h5_benchmarks.py::test_h5_index_many_keys[sorted_fancy]",
            "value": 5.58540330955,
            "unit": "iter/sec",
            "range": "stddev: 0.000245274866853",
            "extra": "mean: 179.038100667 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/storage/test_h5_benchmarks.py::test_h5_index_many_keys[unsorted_fancy]",
            "value": 5.57503662284,
            "unit": "iter/sec",
            "range": "stddev: 0.000277048268045",
            "extra": "mean: 179.371019 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/storage/test_h5_benchmarks.py::test_h5_index_large_arrays[slice]",
            "value": 205.791325788,
            "unit": "iter/sec",
            "range": "stddev: 0.0000301755905796",
            "extra": "mean: 4.85929130478 msec\nrounds: 187"
          },
          {
            "name": "benchmarks/storage/test_h5_benchmarks.py::test_h5_index_large_arrays[sorted_fancy]",
            "value": 38.5115458475,
            "unit": "iter/sec",
            "range": "stddev: 0.000103858338107",
            "extra": "mean: 25.966238903 msec\nrounds: 31"
          },
          {
            "name": "benchmarks/storage/test_h5_benchmarks.py::test_h5_index_large_arrays[dense_fancy]",
            "value": 9.75024786196,
            "unit": "iter/sec",
            "range": "stddev: 0.00466956853984",
            "extra": "mean: 102.561495272 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/storage/test_h5_benchmarks.py::test_h5_index_large_arrays[dense_mask]",
            "value": 9.83175672849,
            "unit": "iter/sec",
            "range": "stddev: 0.00225954220953",
            "extra": "mean: 101.711222889 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/storage/test_h5_benchmarks.py::test_h5_to_tensordict_many_keys",
            "value": 12.0419926803,
            "unit": "iter/sec",
            "range": "stddev: 0.000191893413918",
            "extra": "mean: 83.0427344167 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/storage/test_h5_benchmarks.py::test_h5_index_string_column[int]",
            "value": 45.1819000022,
            "unit": "iter/sec",
            "range": "stddev: 0.00025015310202",
            "extra": "mean: 22.1327567001 msec\nrounds: 40"
          },
          {
            "name": "benchmarks/storage/test_h5_benchmarks.py::test_h5_index_string_column[slice]",
            "value": 45.2104288847,
            "unit": "iter/sec",
            "range": "stddev: 0.0000742872306713",
            "extra": "mean: 22.1187903913 msec\nrounds: 46"
          },
          {
            "name": "benchmarks/storage/test_h5_benchmarks.py::test_h5_index_string_column[sorted_fancy]",
            "value": 45.0093562748,
            "unit": "iter/sec",
            "range": "stddev: 0.0000786807059132",
            "extra": "mean: 22.2176028001 msec\nrounds: 45"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init",
            "value": 66030.0611613,
            "unit": "iter/sec",
            "range": "stddev: 7.90864482486e-7",
            "extra": "mean: 15.1446172003 usec\nrounds: 5606"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_only",
            "value": 155121.453773,
            "unit": "iter/sec",
            "range": "stddev: 4.72375015952e-7",
            "extra": "mean: 6.44656155338 usec\nrounds: 21022"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_ttd_init",
            "value": 285474.194031,
            "unit": "iter/sec",
            "range": "stddev: 3.89630445133e-7",
            "extra": "mean: 3.50294359668 usec\nrounds: 28209"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_td_init_reference",
            "value": 526248.124515,
            "unit": "iter/sec",
            "range": "stddev: 2.89155971173e-7",
            "extra": "mean: 1.90024430191 usec\nrounds: 90580"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_nested",
            "value": 32669.1739902,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013032050872",
            "extra": "mean: 30.6098954417 usec\nrounds: 5175"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size0-False-4]",
            "value": 197749.550351,
            "unit": "iter/sec",
            "range": "stddev: 5.27043517313e-7",
            "extra": "mean: 5.05690151116 usec\nrounds: 45517"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size0-False-16]",
            "value": 122790.591766,
            "unit": "iter/sec",
            "range": "stddev: 6.51838753621e-7",
            "extra": "mean: 8.14394641818 usec\nrounds: 42123"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size0-False-64]",
            "value": 50054.1324723,
            "unit": "iter/sec",
            "range": "stddev: 9.68233173353e-7",
            "extra": "mean: 19.9783704283 usec\nrounds: 27398"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size0-False-256]",
            "value": 14258.1603049,
            "unit": "iter/sec",
            "range": "stddev: 0.00000208159173555",
            "extra": "mean: 70.1352754222 usec\nrounds: 10522"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size0-True-4]",
            "value": 196804.362729,
            "unit": "iter/sec",
            "range": "stddev: 4.5382248743e-7",
            "extra": "mean: 5.08118817152 usec\nrounds: 50917"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size0-True-16]",
            "value": 123220.69887,
            "unit": "iter/sec",
            "range": "stddev: 6.18494808861e-7",
            "extra": "mean: 8.11551962591 usec\nrounds: 44763"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size0-True-64]",
            "value": 50050.6253884,
            "unit": "iter/sec",
            "range": "stddev: 9.27294951849e-7",
            "extra": "mean: 19.9797703274 usec\nrounds: 25833"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size0-True-256]",
            "value": 14238.1800626,
            "unit": "iter/sec",
            "range": "stddev: 0.00000191353640334",
            "extra": "mean: 70.2336952901 usec\nrounds: 10610"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size1-False-4]",
            "value": 59942.3280741,
            "unit": "iter/sec",
            "range": "stddev: 9.24309191563e-7",
            "extra": "mean: 16.6827020593 usec\nrounds: 22988"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size1-False-16]",
            "value": 18791.2656235,
            "unit": "iter/sec",
            "range": "stddev: 0.00000168224079019",
            "extra": "mean: 53.2162133214 usec\nrounds: 12826"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size1-False-64]",
            "value": 5081.95031119,
            "unit": "iter/sec",
            "range": "stddev: 0.00000301159574236",
            "extra": "mean: 196.774847994 usec\nrounds: 4342"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size1-False-256]",
            "value": 1294.93196352,
            "unit": "iter/sec",
            "range": "stddev: 0.00000773164617835",
            "extra": "mean: 772.241344082 usec\nrounds: 1212"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size1-True-4]",
            "value": 59904.6898418,
            "unit": "iter/sec",
            "range": "stddev: 9.33197747027e-7",
            "extra": "mean: 16.6931838332 usec\nrounds: 25051"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size1-True-16]",
            "value": 18832.8526135,
            "unit": "iter/sec",
            "range": "stddev: 0.00000161847946544",
            "extra": "mean: 53.0987004742 usec\nrounds: 12169"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size1-True-64]",
            "value": 5122.31199101,
            "unit": "iter/sec",
            "range": "stddev: 0.00000494584281306",
            "extra": "mean: 195.22434435 usec\nrounds: 3752"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size1-True-256]",
            "value": 1284.80875202,
            "unit": "iter/sec",
            "range": "stddev: 0.0000065444443823",
            "extra": "mean: 778.32595585 usec\nrounds: 1223"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size0-False-4]",
            "value": 77186.6598621,
            "unit": "iter/sec",
            "range": "stddev: 9.1681371361e-7",
            "extra": "mean: 12.9556066008 usec\nrounds: 17829"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size0-False-16]",
            "value": 31749.9004604,
            "unit": "iter/sec",
            "range": "stddev: 0.00000145579284231",
            "extra": "mean: 31.4961617359 usec\nrounds: 11605"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size0-False-64]",
            "value": 9307.41332717,
            "unit": "iter/sec",
            "range": "stddev: 0.00000264639986056",
            "extra": "mean: 107.441236877 usec\nrounds: 7223"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size0-False-256]",
            "value": 2502.38303911,
            "unit": "iter/sec",
            "range": "stddev: 0.00000593893167594",
            "extra": "mean: 399.619076844 usec\nrounds: 2199"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size0-True-4]",
            "value": 77074.2453263,
            "unit": "iter/sec",
            "range": "stddev: 0.00000122957846574",
            "extra": "mean: 12.974502647 usec\nrounds: 23941"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size0-True-16]",
            "value": 31442.1265469,
            "unit": "iter/sec",
            "range": "stddev: 0.00000134302494588",
            "extra": "mean: 31.8044645773 usec\nrounds: 16122"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size0-True-64]",
            "value": 9383.4637711,
            "unit": "iter/sec",
            "range": "stddev: 0.00000265318220279",
            "extra": "mean: 106.57045462 usec\nrounds: 7151"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size0-True-256]",
            "value": 2482.21354755,
            "unit": "iter/sec",
            "range": "stddev: 0.0000057777047631",
            "extra": "mean: 402.866224377 usec\nrounds: 2224"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size1-False-4]",
            "value": 41076.8888125,
            "unit": "iter/sec",
            "range": "stddev: 0.00000198583520029",
            "extra": "mean: 24.3445895955 usec\nrounds: 15560"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size1-False-16]",
            "value": 13151.7996162,
            "unit": "iter/sec",
            "range": "stddev: 0.00000219914611971",
            "extra": "mean: 76.0352217325 usec\nrounds: 8632"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size1-False-64]",
            "value": 3527.18607894,
            "unit": "iter/sec",
            "range": "stddev: 0.00000411620157723",
            "extra": "mean: 283.512119185 usec\nrounds: 3079"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size1-False-256]",
            "value": 891.182541734,
            "unit": "iter/sec",
            "range": "stddev: 0.0000135547900373",
            "extra": "mean: 1.12210456688 msec\nrounds: 852"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size1-True-4]",
            "value": 41122.1939364,
            "unit": "iter/sec",
            "range": "stddev: 0.00000115884251276",
            "extra": "mean: 24.3177686859 usec\nrounds: 18162"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size1-True-16]",
            "value": 12805.2287824,
            "unit": "iter/sec",
            "range": "stddev: 0.00000216490527668",
            "extra": "mean: 78.0930990762 usec\nrounds: 8610"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size1-True-64]",
            "value": 3531.94622253,
            "unit": "iter/sec",
            "range": "stddev: 0.00000410349674901",
            "extra": "mean: 283.130018691 usec\nrounds: 2886"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size1-True-256]",
            "value": 896.28160341,
            "unit": "iter/sec",
            "range": "stddev: 0.0000106655448623",
            "extra": "mean: 1.11572076923 msec\nrounds: 832"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_nested_tensor_schema[False-False-4]",
            "value": 7896.09310108,
            "unit": "iter/sec",
            "range": "stddev: 0.00000310310680589",
            "extra": "mean: 126.644909983 usec\nrounds: 4566"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_nested_tensor_schema[False-False-64]",
            "value": 528.157205738,
            "unit": "iter/sec",
            "range": "stddev: 0.0000125343160393",
            "extra": "mean: 1.89337566379 msec\nrounds: 473"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_nested_tensor_schema[False-True-4]",
            "value": 8429.13312394,
            "unit": "iter/sec",
            "range": "stddev: 0.00000324671080014",
            "extra": "mean: 118.636161667 usec\nrounds: 5369"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_nested_tensor_schema[False-True-64]",
            "value": 565.074941596,
            "unit": "iter/sec",
            "range": "stddev: 0.0000114115300622",
            "extra": "mean: 1.76967677451 msec\nrounds: 510"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_nested_tensor_schema[True-False-4]",
            "value": 44384.0514247,
            "unit": "iter/sec",
            "range": "stddev: 0.00000110825722993",
            "extra": "mean: 22.530615568 usec\nrounds: 12702"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_nested_tensor_schema[True-False-64]",
            "value": 3914.21543292,
            "unit": "iter/sec",
            "range": "stddev: 0.00000387606761149",
            "extra": "mean: 255.479039705 usec\nrounds: 3149"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_nested_tensor_schema[True-True-4]",
            "value": 63592.2307671,
            "unit": "iter/sec",
            "range": "stddev: 9.1432133434e-7",
            "extra": "mean: 15.7251913944 usec\nrounds: 28051"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_nested_tensor_schema[True-True-64]",
            "value": 6904.55869609,
            "unit": "iter/sec",
            "range": "stddev: 0.00000264730099831",
            "extra": "mean: 144.83184864 usec\nrounds: 5649"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_many_fields",
            "value": 148298.946144,
            "unit": "iter/sec",
            "range": "stddev: 5.90642413792e-7",
            "extra": "mean: 6.74313625283 usec\nrounds: 26575"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_first_layer_tensor",
            "value": 2377027.15908,
            "unit": "iter/sec",
            "range": "stddev: 1.59849391771e-7",
            "extra": "mean: 420.693552524 nsec\nrounds: 161551"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_first_layer_tensor_only",
            "value": 4988269.36737,
            "unit": "iter/sec",
            "range": "stddev: 1.95240176726e-8",
            "extra": "mean: 200.470328756 nsec\nrounds: 197239"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_ttd_first_layer_tensor",
            "value": 2143766.60653,
            "unit": "iter/sec",
            "range": "stddev: 1.47105025752e-7",
            "extra": "mean: 466.468689715 nsec\nrounds: 89929"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_td_first_layer_tensor_reference",
            "value": 4879109.4553,
            "unit": "iter/sec",
            "range": "stddev: 1.98354482285e-8",
            "extra": "mean: 204.955434831 nsec\nrounds: 191571"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_first_layer_tensor_set",
            "value": 497181.512532,
            "unit": "iter/sec",
            "range": "stddev: 3.31528262613e-7",
            "extra": "mean: 2.01133786111 usec\nrounds: 110742"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_first_layer_tensor_only_set",
            "value": 876774.160246,
            "unit": "iter/sec",
            "range": "stddev: 2.16640312489e-7",
            "extra": "mean: 1.14054456135 usec\nrounds: 98329"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_first_layer_nontensor",
            "value": 723425.019946,
            "unit": "iter/sec",
            "range": "stddev: 2.33795030322e-7",
            "extra": "mean: 1.3823132632 usec\nrounds: 87413"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_second_layer_tensor",
            "value": 1759732.67945,
            "unit": "iter/sec",
            "range": "stddev: 3.46226376558e-8",
            "extra": "mean: 568.26813054 nsec\nrounds: 77882"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_second_layer_nontensor",
            "value": 609005.111004,
            "unit": "iter/sec",
            "range": "stddev: 2.41724364409e-7",
            "extra": "mean: 1.6420223442 usec\nrounds: 141644"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_clone",
            "value": 55672.0588605,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012708468468",
            "extra": "mean: 17.9623319214 usec\nrounds: 11720"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_nontensordata_init",
            "value": 242372.246452,
            "unit": "iter/sec",
            "range": "stddev: 4.598542292e-7",
            "extra": "mean: 4.12588493377 usec\nrounds: 39448"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_nontensordata_data",
            "value": 1194440.62435,
            "unit": "iter/sec",
            "range": "stddev: 1.99269157611e-7",
            "extra": "mean: 837.211979914 nsec\nrounds: 110133"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_nontensordata_copy",
            "value": 350094.863303,
            "unit": "iter/sec",
            "range": "stddev: 3.72348640044e-7",
            "extra": "mean: 2.85636867267 usec\nrounds: 52302"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_nontensor_lazy_stack_copy",
            "value": 89.0916689811,
            "unit": "iter/sec",
            "range": "stddev: 0.0503460501046",
            "extra": "mean: 11.2243940588 msec\nrounds: 153"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_nontensor_stack_of_dicts_copy",
            "value": 916.153848681,
            "unit": "iter/sec",
            "range": "stddev: 0.00000998113009691",
            "extra": "mean: 1.09151972831 msec\nrounds: 865"
          },
          {
            "name": "benchmarks/tensorclass/test_torch_functions.py::test_unbind",
            "value": 72.0054511041,
            "unit": "iter/sec",
            "range": "stddev: 0.060268799356",
            "extra": "mean: 13.8878374438 msec\nrounds: 160"
          },
          {
            "name": "benchmarks/tensorclass/test_torch_functions.py::test_full_like",
            "value": 2907.23446919,
            "unit": "iter/sec",
            "range": "stddev: 0.000112713895355",
            "extra": "mean: 343.969504557 usec\nrounds: 1096"
          },
          {
            "name": "benchmarks/tensorclass/test_torch_functions.py::test_zeros_like",
            "value": 3074.18468782,
            "unit": "iter/sec",
            "range": "stddev: 0.0000829581254357",
            "extra": "mean: 325.289499998 usec\nrounds: 1186"
          },
          {
            "name": "benchmarks/tensorclass/test_torch_functions.py::test_ones_like",
            "value": 2490.43116693,
            "unit": "iter/sec",
            "range": "stddev: 0.000128119950529",
            "extra": "mean: 401.536895811 usec\nrounds: 1507"
          },
          {
            "name": "benchmarks/tensorclass/test_torch_functions.py::test_clone",
            "value": 2282.89002578,
            "unit": "iter/sec",
            "range": "stddev: 0.000091440968277",
            "extra": "mean: 438.04124978 usec\nrounds: 1233"
          },
          {
            "name": "benchmarks/tensorclass/test_torch_functions.py::test_squeeze",
            "value": 116177.818193,
            "unit": "iter/sec",
            "range": "stddev: 8.39839986437e-7",
            "extra": "mean: 8.60749509288 usec\nrounds: 18245"
          },
          {
            "name": "benchmarks/tensorclass/test_torch_functions.py::test_unsqueeze",
            "value": 19614.9071828,
            "unit": "iter/sec",
            "range": "stddev: 0.00000217457760494",
            "extra": "mean: 50.9816330346 usec\nrounds: 4480"
          },
          {
            "name": "benchmarks/tensorclass/test_torch_functions.py::test_split",
            "value": 10904.6498675,
            "unit": "iter/sec",
            "range": "stddev: 0.00000286661402059",
            "extra": "mean: 91.7039989497 usec\nrounds: 3880"
          },
          {
            "name": "benchmarks/tensorclass/test_torch_functions.py::test_permute",
            "value": 15232.5349394,
            "unit": "iter/sec",
            "range": "stddev: 0.00000225544070162",
            "extra": "mean: 65.6489549492 usec\nrounds: 4507"
          },
          {
            "name": "benchmarks/tensorclass/test_torch_functions.py::test_stack",
            "value": 341.405035634,
            "unit": "iter/sec",
            "range": "stddev: 0.000405430536662",
            "extra": "mean: 2.92907220347 msec\nrounds: 300"
          },
          {
            "name": "benchmarks/tensorclass/test_torch_functions.py::test_cat",
            "value": 95.4004004368,
            "unit": "iter/sec",
            "range": "stddev: 0.000591724675519",
            "extra": "mean: 10.4821362952 msec\nrounds: 315"
          },
          {
            "name": "benchmarks/test_fx_benchmarks.py::TestFxSequential::test_sequential_tensordict",
            "value": 4604.69318529,
            "unit": "iter/sec",
            "range": "stddev: 0.000035829798897",
            "extra": "mean: 217.169735259 usec\nrounds: 1938"
          },
          {
            "name": "benchmarks/test_fx_benchmarks.py::TestFxSequential::test_sequential_graph_module",
            "value": 5575.82366726,
            "unit": "iter/sec",
            "range": "stddev: 0.0000140337839552",
            "extra": "mean: 179.345700236 usec\nrounds: 3099"
          },
          {
            "name": "benchmarks/test_fx_benchmarks.py::TestFxNested::test_nested_tensordict",
            "value": 3444.37602656,
            "unit": "iter/sec",
            "range": "stddev: 0.0000212988942002",
            "extra": "mean: 290.328347512 usec\nrounds: 2279"
          },
          {
            "name": "benchmarks/test_fx_benchmarks.py::TestFxNested::test_nested_graph_module",
            "value": 4233.42068118,
            "unit": "iter/sec",
            "range": "stddev: 0.0000156263944388",
            "extra": "mean: 236.215598522 usec\nrounds: 2802"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "peterdsharpe@gmail.com",
            "name": "Peter Sharpe",
            "username": "peterdsharpe"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "69270737b4b74fd280dc89c4eca9aa45c3d5ac55",
          "message": "[CI] Fail the docs build on Sphinx warnings, and build it on an 8-core runner instead of a 48-core one (#3057)",
          "timestamp": "2026-10-19T11:14:53-04:00",
          "tree_id": "81fd6f5a68d9b4e636a9a425ec795a459628f242",
          "url": "https://github.com/pytorch/tensordict/commit/69270737b4b74fd280dc89c4eca9aa45c3d5ac55"
        },
        "date": 1792423876852,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cpu-32-2]",
            "value": 14514.6714373,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025703162826",
            "extra": "mean: 68.8958068615 usec\nrounds: 3935"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cpu-32-8]",
            "value": 9610.26703279,
            "unit": "iter/sec",
            "range": "stddev: 0.00000299424612007",
            "extra": "mean: 104.055381249 usec\nrounds: 5781"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cpu-32-32]",
            "value": 3854.30962364,
            "unit": "iter/sec",
            "range": "stddev: 0.00000493664211034",
            "extra": "mean: 259.449836066 usec\nrounds: 2684"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cpu-4096-2]",
            "value": 4966.3581614,
            "unit": "iter/sec",
            "range": "stddev: 0.000183671805067",
            "extra": "mean: 201.354789063 usec\nrounds: 1024"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cpu-4096-8]",
            "value": 1514.06499248,
            "unit": "iter/sec",
            "range": "stddev: 0.00170960545157",
            "extra": "mean: 660.473628916 usec\nrounds: 415"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cpu-4096-32]",
            "value": 347.456344126,
            "unit": "iter/sec",
            "range": "stddev: 0.0000583324866613",
            "extra": "mean: 2.87805940777 msec\nrounds: 103"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cpu-32-2]",
            "value": 7676.78248887,
            "unit": "iter/sec",
            "range": "stddev: 0.00000297949203735",
            "extra": "mean: 130.262906556 usec\nrounds: 2654"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cpu-32-8]",
            "value": 1934.48259039,
            "unit": "iter/sec",
            "range": "stddev: 0.00000795742871545",
            "extra": "mean: 516.934091299 usec\nrounds: 1632"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cpu-32-32]",
            "value": 483.878299492,
            "unit": "iter/sec",
            "range": "stddev: 0.0000130518326977",
            "extra": "mean: 2.06663535242 msec\nrounds: 454"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cpu-4096-2]",
            "value": 4171.97868835,
            "unit": "iter/sec",
            "range": "stddev: 0.0000178787902207",
            "extra": "mean: 239.694417134 usec\nrounds: 1249"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cpu-4096-8]",
            "value": 973.521568163,
            "unit": "iter/sec",
            "range": "stddev: 0.0000447176894668",
            "extra": "mean: 1.02719860833 msec\nrounds: 360"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cpu-4096-32]",
            "value": 195.096492282,
            "unit": "iter/sec",
            "range": "stddev: 0.0000658479635289",
            "extra": "mean: 5.12566878218 msec\nrounds: 101"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cpu-32-2]",
            "value": 12196.2584532,
            "unit": "iter/sec",
            "range": "stddev: 0.00000232926856183",
            "extra": "mean: 81.9923588729 usec\nrounds: 4542"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cpu-32-8]",
            "value": 3018.85044524,
            "unit": "iter/sec",
            "range": "stddev: 0.00000681400717889",
            "extra": "mean: 331.251917953 usec\nrounds: 2462"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cpu-32-32]",
            "value": 755.99832614,
            "unit": "iter/sec",
            "range": "stddev: 0.0000173409816365",
            "extra": "mean: 1.32275425146 msec\nrounds: 684"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cpu-4096-2]",
            "value": 4983.82492293,
            "unit": "iter/sec",
            "range": "stddev: 0.0000149288188679",
            "extra": "mean: 200.649102941 usec\nrounds: 1360"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cpu-4096-8]",
            "value": 1133.62421734,
            "unit": "iter/sec",
            "range": "stddev: 0.000227032561402",
            "extra": "mean: 882.1265325 usec\nrounds: 400"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cpu-4096-32]",
            "value": 225.424756598,
            "unit": "iter/sec",
            "range": "stddev: 0.0000666581641156",
            "extra": "mean: 4.43607 msec\nrounds: 105"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cpu-32-2]",
            "value": 16768.095485,
            "unit": "iter/sec",
            "range": "stddev: 0.00000415959223415",
            "extra": "mean: 59.6370649782 usec\nrounds: 7818"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cpu-32-8]",
            "value": 4253.46388264,
            "unit": "iter/sec",
            "range": "stddev: 0.00000488009405267",
            "extra": "mean: 235.102501771 usec\nrounds: 3671"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cpu-32-32]",
            "value": 1034.21605655,
            "unit": "iter/sec",
            "range": "stddev: 0.00000712167334397",
            "extra": "mean: 966.915949206 usec\nrounds: 945"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cpu-4096-2]",
            "value": 5878.09366329,
            "unit": "iter/sec",
            "range": "stddev: 0.000125492305312",
            "extra": "mean: 170.123182324 usec\nrounds: 1437"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cpu-4096-8]",
            "value": 1402.70854208,
            "unit": "iter/sec",
            "range": "stddev: 0.0000190467342559",
            "extra": "mean: 712.906473441 usec\nrounds: 433"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cpu-4096-32]",
            "value": 262.324633759,
            "unit": "iter/sec",
            "range": "stddev: 0.0000623555188301",
            "extra": "mean: 3.81207050847 msec\nrounds: 118"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cpu-128-128]",
            "value": 218.016820965,
            "unit": "iter/sec",
            "range": "stddev: 0.0000806741632433",
            "extra": "mean: 4.58680204386 msec\nrounds: 114"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cpu-128-256]",
            "value": 94.7814108452,
            "unit": "iter/sec",
            "range": "stddev: 0.0000996367664416",
            "extra": "mean: 10.5505920526 msec\nrounds: 76"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cpu-4096-128]",
            "value": 14.8800142821,
            "unit": "iter/sec",
            "range": "stddev: 0.000656681487027",
            "extra": "mean: 67.2042365714 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cpu-4096-256]",
            "value": 6.60000777613,
            "unit": "iter/sec",
            "range": "stddev: 0.000760050738479",
            "extra": "mean: 151.514973 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cpu-128-128]",
            "value": 34.8016106599,
            "unit": "iter/sec",
            "range": "stddev: 0.0000674683600019",
            "extra": "mean: 28.7343022647 msec\nrounds: 34"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cpu-128-256]",
            "value": 16.5744825795,
            "unit": "iter/sec",
            "range": "stddev: 0.000203381026446",
            "extra": "mean: 60.3337084706 msec\nrounds: 17"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cpu-4096-128]",
            "value": 8.08901340442,
            "unit": "iter/sec",
            "range": "stddev: 0.00068869176263",
            "extra": "mean: 123.624470625 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cpu-4096-256]",
            "value": 3.8685808545,
            "unit": "iter/sec",
            "range": "stddev: 0.00201057697153",
            "extra": "mean: 258.4927232 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cpu-128-128]",
            "value": 61.0951265836,
            "unit": "iter/sec",
            "range": "stddev: 0.000589311728402",
            "extra": "mean: 16.3679176379 msec\nrounds: 58"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cpu-128-256]",
            "value": 28.0417497348,
            "unit": "iter/sec",
            "range": "stddev: 0.000315020035254",
            "extra": "mean: 35.6611127857 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cpu-4096-128]",
            "value": 8.399419758,
            "unit": "iter/sec",
            "range": "stddev: 0.00583605464605",
            "extra": "mean: 119.055843 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cpu-4096-256]",
            "value": 4.30499281598,
            "unit": "iter/sec",
            "range": "stddev: 0.0122244186298",
            "extra": "mean: 232.2884248 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cpu-128-128]",
            "value": 98.5977255328,
            "unit": "iter/sec",
            "range": "stddev: 0.0000422940526989",
            "extra": "mean: 10.1422217865 msec\nrounds: 89"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cpu-128-256]",
            "value": 44.3374438715,
            "unit": "iter/sec",
            "range": "stddev: 0.00011690569775",
            "extra": "mean: 22.5542997674 msec\nrounds: 43"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cpu-4096-128]",
            "value": 10.2838951817,
            "unit": "iter/sec",
            "range": "stddev: 0.00065786421337",
            "extra": "mean: 97.2394197273 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cpu-4096-256]",
            "value": 4.79665880711,
            "unit": "iter/sec",
            "range": "stddev: 0.000832494872037",
            "extra": "mean: 208.4784514 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[None-batch_size0-4]",
            "value": 536891.976559,
            "unit": "iter/sec",
            "range": "stddev: 2.91083012278e-7",
            "extra": "mean: 1.86257206973 usec\nrounds: 54378"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[None-batch_size0-16]",
            "value": 335342.077857,
            "unit": "iter/sec",
            "range": "stddev: 3.28808360996e-7",
            "extra": "mean: 2.98202959316 usec\nrounds: 96340"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[None-batch_size0-64]",
            "value": 129632.74031,
            "unit": "iter/sec",
            "range": "stddev: 5.81243382386e-7",
            "extra": "mean: 7.71410060152 usec\nrounds: 51520"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[None-batch_size0-256]",
            "value": 37256.7741964,
            "unit": "iter/sec",
            "range": "stddev: 0.00000110481855815",
            "extra": "mean: 26.8407563878 usec\nrounds: 26731"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[None-batch_size1-4]",
            "value": 299515.225904,
            "unit": "iter/sec",
            "range": "stddev: 3.7958231124e-7",
            "extra": "mean: 3.33872843018 usec\nrounds: 66841"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[None-batch_size1-16]",
            "value": 117766.959477,
            "unit": "iter/sec",
            "range": "stddev: 6.00930062746e-7",
            "extra": "mean: 8.49134599754 usec\nrounds: 57538"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[None-batch_size1-64]",
            "value": 33807.7163162,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011600887172",
            "extra": "mean: 29.5790461162 usec\nrounds: 25284"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[None-batch_size1-256]",
            "value": 9168.80442445,
            "unit": "iter/sec",
            "range": "stddev: 0.00000349436890344",
            "extra": "mean: 109.065473938 usec\nrounds: 7252"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[cpu-batch_size0-4]",
            "value": 323023.0492,
            "unit": "iter/sec",
            "range": "stddev: 4.63310831909e-7",
            "extra": "mean: 3.09575431993 usec\nrounds: 48380"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[cpu-batch_size0-16]",
            "value": 192716.642828,
            "unit": "iter/sec",
            "range": "stddev: 4.81822745094e-7",
            "extra": "mean: 5.18896544338 usec\nrounds: 58310"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[cpu-batch_size0-64]",
            "value": 71916.6937387,
            "unit": "iter/sec",
            "range": "stddev: 7.68917677034e-7",
            "extra": "mean: 13.9049773844 usec\nrounds: 37231"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[cpu-batch_size0-256]",
            "value": 20475.9141556,
            "unit": "iter/sec",
            "range": "stddev: 0.00000154990994199",
            "extra": "mean: 48.8378683561 usec\nrounds: 15390"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[cpu-batch_size1-4]",
            "value": 217410.71847,
            "unit": "iter/sec",
            "range": "stddev: 6.61363801438e-7",
            "extra": "mean: 4.59958923386 usec\nrounds: 49359"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[cpu-batch_size1-16]",
            "value": 91680.8114182,
            "unit": "iter/sec",
            "range": "stddev: 6.87653163198e-7",
            "extra": "mean: 10.9074078265 usec\nrounds: 39762"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[cpu-batch_size1-64]",
            "value": 27305.474739,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012839600191",
            "extra": "mean: 36.6226923193 usec\nrounds: 19608"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_construct_tensor_leaves[cpu-batch_size1-256]",
            "value": 7301.682199,
            "unit": "iter/sec",
            "range": "stddev: 0.00000263741112286",
            "extra": "mean: 136.954741763 usec\nrounds: 6738"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[constructor-dict-16]",
            "value": 39196.9876636,
            "unit": "iter/sec",
            "range": "stddev: 0.00000114938990018",
            "extra": "mean: 25.512164572 usec\nrounds: 11138"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[constructor-dict-256]",
            "value": 2724.7930602,
            "unit": "iter/sec",
            "range": "stddev: 0.0000043272841409",
            "extra": "mean: 367.000347515 usec\nrounds: 2374"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[constructor-tuple_keys-16]",
            "value": 27685.6519131,
            "unit": "iter/sec",
            "range": "stddev: 0.00000136793583819",
            "extra": "mean: 36.1197924159 usec\nrounds: 14399"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[constructor-tuple_keys-256]",
            "value": 1847.64005823,
            "unit": "iter/sec",
            "range": "stddev: 0.00000568942423009",
            "extra": "mean: 541.230958673 usec\nrounds: 1718"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[constructor-ready_children-16]",
            "value": 138691.492075,
            "unit": "iter/sec",
            "range": "stddev: 5.16158076343e-7",
            "extra": "mean: 7.21024761532 usec\nrounds: 64475"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[constructor-ready_children-256]",
            "value": 14383.8796622,
            "unit": "iter/sec",
            "range": "stddev: 0.00000233861320373",
            "extra": "mean: 69.5222723968 usec\nrounds: 12937"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[set-dict-16]",
            "value": 35988.1737243,
            "unit": "iter/sec",
            "range": "stddev: 0.00000140523518659",
            "extra": "mean: 27.7869059892 usec\nrounds: 14977"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[set-dict-256]",
            "value": 2364.87196941,
            "unit": "iter/sec",
            "range": "stddev: 0.00000497703838105",
            "extra": "mean: 422.85587251 usec\nrounds: 1004"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[set-tuple_keys-16]",
            "value": 30144.9093135,
            "unit": "iter/sec",
            "range": "stddev: 0.00000143416226707",
            "extra": "mean: 33.1730969763 usec\nrounds: 16767"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[set-tuple_keys-256]",
            "value": 1978.63694785,
            "unit": "iter/sec",
            "range": "stddev: 0.0000136854566707",
            "extra": "mean: 505.398426471 usec\nrounds: 1836"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[set-ready_children-16]",
            "value": 114173.762982,
            "unit": "iter/sec",
            "range": "stddev: 6.24083619587e-7",
            "extra": "mean: 8.75857967611 usec\nrounds: 42177"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[set-ready_children-256]",
            "value": 8821.09188308,
            "unit": "iter/sec",
            "range": "stddev: 0.00000258668779813",
            "extra": "mean: 113.364650687 usec\nrounds: 8156"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[update-dict-16]",
            "value": 31502.9495625,
            "unit": "iter/sec",
            "range": "stddev: 0.00000136747927397",
            "extra": "mean: 31.743059424 usec\nrounds: 11359"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[update-dict-256]",
            "value": 2140.56742248,
            "unit": "iter/sec",
            "range": "stddev: 0.00000579075769042",
            "extra": "mean: 467.165850278 usec\nrounds: 1082"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[update-tuple_keys-16]",
            "value": 23653.5959406,
            "unit": "iter/sec",
            "range": "stddev: 0.00000165016421711",
            "extra": "mean: 42.276869974 usec\nrounds: 13405"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[update-tuple_keys-256]",
            "value": 1571.04245843,
            "unit": "iter/sec",
            "range": "stddev: 0.00000609436137814",
            "extra": "mean: 636.520034603 usec\nrounds: 1445"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[update-ready_children-16]",
            "value": 79309.6729719,
            "unit": "iter/sec",
            "range": "stddev: 9.06952751938e-7",
            "extra": "mean: 12.6088024642 usec\nrounds: 32627"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_build_nested_tensor_leaves[update-ready_children-256]",
            "value": 6465.4071559,
            "unit": "iter/sec",
            "range": "stddev: 0.00000311201001522",
            "extra": "mean: 154.669300152 usec\nrounds: 5927"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 117701.9738,
            "unit": "iter/sec",
            "range": "stddev: 6.91036118956e-7",
            "extra": "mean: 8.4960342441 usec\nrounds: 21259"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 117631.997069,
            "unit": "iter/sec",
            "range": "stddev: 7.78901247136e-7",
            "extra": "mean: 8.50108835111 usec\nrounds: 19332"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 115172.594983,
            "unit": "iter/sec",
            "range": "stddev: 7.90852994577e-7",
            "extra": "mean: 8.68262107099 usec\nrounds: 61051"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 114919.155169,
            "unit": "iter/sec",
            "range": "stddev: 7.19239286837e-7",
            "extra": "mean: 8.70176950508 usec\nrounds: 76570"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_td_batchfree",
            "value": 1389010.28445,
            "unit": "iter/sec",
            "range": "stddev: 1.81170175798e-7",
            "extra": "mean: 719.937074043 nsec\nrounds: 197629"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 293868.565041,
            "unit": "iter/sec",
            "range": "stddev: 3.93767380248e-7",
            "extra": "mean: 3.40288182868 usec\nrounds: 90335"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 4386.77884554,
            "unit": "iter/sec",
            "range": "stddev: 0.00000440646463617",
            "extra": "mean: 227.957696344 usec\nrounds: 3392"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 4351.63681262,
            "unit": "iter/sec",
            "range": "stddev: 0.00000406745967143",
            "extra": "mean: 229.798589142 usec\nrounds: 4089"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 17871.2916044,
            "unit": "iter/sec",
            "range": "stddev: 0.00000191324504971",
            "extra": "mean: 55.9556646568 usec\nrounds: 10783"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 4551.262283,
            "unit": "iter/sec",
            "range": "stddev: 0.00000446298325565",
            "extra": "mean: 219.7192642 usec\nrounds: 3838"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 18276.794113,
            "unit": "iter/sec",
            "range": "stddev: 0.00000192314474897",
            "extra": "mean: 54.7141907829 usec\nrounds: 13822"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 4425.44539417,
            "unit": "iter/sec",
            "range": "stddev: 0.00000461801627402",
            "extra": "mean: 225.965956176 usec\nrounds: 4153"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 880417.154752,
            "unit": "iter/sec",
            "range": "stddev: 2.37635019603e-7",
            "extra": "mean: 1.13582521036 usec\nrounds: 162338"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 38948.0954393,
            "unit": "iter/sec",
            "range": "stddev: 0.00000136130865237",
            "extra": "mean: 25.6751964048 usec\nrounds: 14964"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 33078.9661416,
            "unit": "iter/sec",
            "range": "stddev: 0.00000505050751761",
            "extra": "mean: 30.2306908783 usec\nrounds: 24864"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 41843.8632043,
            "unit": "iter/sec",
            "range": "stddev: 0.00000162397031242",
            "extra": "mean: 23.8983670107 usec\nrounds: 24438"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf_loop",
            "value": 71829.712969,
            "unit": "iter/sec",
            "range": "stddev: 9.95988570317e-7",
            "extra": "mean: 13.9218153417 usec\nrounds: 43085"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf_small",
            "value": 237284.274028,
            "unit": "iter/sec",
            "range": "stddev: 4.47149059498e-7",
            "extra": "mean: 4.21435429759 usec\nrounds: 74683"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf_small_len",
            "value": 388639.572545,
            "unit": "iter/sec",
            "range": "stddev: 3.80236802064e-7",
            "extra": "mean: 2.57307816971 usec\nrounds: 108226"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 39241.2114295,
            "unit": "iter/sec",
            "range": "stddev: 0.00000137674564668",
            "extra": "mean: 25.4834130643 usec\nrounds: 22994"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 42531.0126947,
            "unit": "iter/sec",
            "range": "stddev: 0.00000119023675102",
            "extra": "mean: 23.5122546265 usec\nrounds: 26533"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 33328.240827,
            "unit": "iter/sec",
            "range": "stddev: 0.00000144562622636",
            "extra": "mean: 30.004583956 usec\nrounds: 26240"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_lazy_stack_nested",
            "value": 4648.33703052,
            "unit": "iter/sec",
            "range": "stddev: 0.00000412653365105",
            "extra": "mean: 215.130700169 usec\nrounds: 3542"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_lazy_stack_nested_leaf",
            "value": 4566.19193208,
            "unit": "iter/sec",
            "range": "stddev: 0.00000406603547522",
            "extra": "mean: 219.000868749 usec\nrounds: 3939"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_lazy_stack_nested_leaf_loop",
            "value": 9036.32915204,
            "unit": "iter/sec",
            "range": "stddev: 0.00000290086656639",
            "extra": "mean: 110.664406218 usec\nrounds: 7720"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 1516834.65747,
            "unit": "iter/sec",
            "range": "stddev: 6.80765347098e-8",
            "extra": "mean: 659.26763677 nsec\nrounds: 174217"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 35502.9760137,
            "unit": "iter/sec",
            "range": "stddev: 0.00000133284326046",
            "extra": "mean: 28.1666528354 usec\nrounds: 19328"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 33777.8542086,
            "unit": "iter/sec",
            "range": "stddev: 0.00000148304280443",
            "extra": "mean: 29.6051961686 usec\nrounds: 29648"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 33476.0032659,
            "unit": "iter/sec",
            "range": "stddev: 0.00000122308926702",
            "extra": "mean: 29.8721442956 usec\nrounds: 13209"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 35663.7469889,
            "unit": "iter/sec",
            "range": "stddev: 0.00000121387328934",
            "extra": "mean: 28.0396785091 usec\nrounds: 25727"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 33272.2183329,
            "unit": "iter/sec",
            "range": "stddev: 0.00000124596160308",
            "extra": "mean: 30.0551045318 usec\nrounds: 24184"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 33335.816837,
            "unit": "iter/sec",
            "range": "stddev: 0.00000127305362832",
            "extra": "mean: 29.9977650132 usec\nrounds: 29291"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 1780939.81473,
            "unit": "iter/sec",
            "range": "stddev: 1.4979449373e-7",
            "extra": "mean: 561.501288099 nsec\nrounds: 145561"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 644266.050178,
            "unit": "iter/sec",
            "range": "stddev: 2.96278837708e-7",
            "extra": "mean: 1.55215380311 usec\nrounds: 71429"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 645157.245579,
            "unit": "iter/sec",
            "range": "stddev: 2.539585763e-7",
            "extra": "mean: 1.55000971756 usec\nrounds: 156007"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 641329.553765,
            "unit": "iter/sec",
            "range": "stddev: 2.86984282638e-7",
            "extra": "mean: 1.55926074844 usec\nrounds: 137742"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 646198.536029,
            "unit": "iter/sec",
            "range": "stddev: 2.81724221027e-7",
            "extra": "mean: 1.5475120172 usec\nrounds: 131062"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 389813.801621,
            "unit": "iter/sec",
            "range": "stddev: 3.93383866367e-7",
            "extra": "mean: 2.5653273328 usec\nrounds: 61051"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 409254.331601,
            "unit": "iter/sec",
            "range": "stddev: 3.88839832098e-7",
            "extra": "mean: 2.44346833444 usec\nrounds: 96619"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 404082.308234,
            "unit": "iter/sec",
            "range": "stddev: 3.56817733509e-7",
            "extra": "mean: 2.47474333724 usec\nrounds: 89526"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 407171.49835,
            "unit": "iter/sec",
            "range": "stddev: 3.80987131534e-7",
            "extra": "mean: 2.45596758136 usec\nrounds: 83964"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 369625.101952,
            "unit": "iter/sec",
            "range": "stddev: 3.55656105219e-7",
            "extra": "mean: 2.70544396124 usec\nrounds: 85985"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 388163.770245,
            "unit": "iter/sec",
            "range": "stddev: 3.48930311864e-7",
            "extra": "mean: 2.57623219027 usec\nrounds: 167225"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 379763.634755,
            "unit": "iter/sec",
            "range": "stddev: 3.91728226959e-7",
            "extra": "mean: 2.63321684459 usec\nrounds: 130891"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 386135.073097,
            "unit": "iter/sec",
            "range": "stddev: 3.81995377976e-7",
            "extra": "mean: 2.58976733706 usec\nrounds: 136240"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 378312.269758,
            "unit": "iter/sec",
            "range": "stddev: 3.85732478832e-7",
            "extra": "mean: 2.64331897202 usec\nrounds: 116145"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 367159.219377,
            "unit": "iter/sec",
            "range": "stddev: 3.85258529608e-7",
            "extra": "mean: 2.7236140269 usec\nrounds: 162602"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 380081.089789,
            "unit": "iter/sec",
            "range": "stddev: 3.9759994751e-7",
            "extra": "mean: 2.63101750354 usec\nrounds: 107864"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 370395.698919,
            "unit": "iter/sec",
            "range": "stddev: 3.73122867827e-7",
            "extra": "mean: 2.69981536751 usec\nrounds: 140647"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_shallow_nested_getitemleaf",
            "value": 1219695.24842,
            "unit": "iter/sec",
            "range": "stddev: 8.14141802088e-8",
            "extra": "mean: 819.876933435 nsec\nrounds: 195313"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_is_non_tensor[tensor]",
            "value": 10375937.4753,
            "unit": "iter/sec",
            "range": "stddev: 6.3753717658e-9",
            "extra": "mean: 96.3768336484 nsec\nrounds: 98912"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_is_non_tensor[parameter]",
            "value": 3501949.42395,
            "unit": "iter/sec",
            "range": "stddev: 1.87482941918e-7",
            "extra": "mean: 285.555237652 nsec\nrounds: 166390"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_is_non_tensor[tensordict]",
            "value": 5345659.77225,
            "unit": "iter/sec",
            "range": "stddev: 8.63530297265e-9",
            "extra": "mean: 187.067647887 nsec\nrounds: 52574"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 4655.10724711,
            "unit": "iter/sec",
            "range": "stddev: 0.00000604409354266",
            "extra": "mean: 214.817822 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 4285.1048948,
            "unit": "iter/sec",
            "range": "stddev: 0.0000388571261003",
            "extra": "mean: 233.366516001 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 4353.42315059,
            "unit": "iter/sec",
            "range": "stddev: 0.0000100507277172",
            "extra": "mean: 229.704296001 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 4029.53234587,
            "unit": "iter/sec",
            "range": "stddev: 0.0000302125114074",
            "extra": "mean: 248.167756 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_data_nested_locked",
            "value": 17742.6673688,
            "unit": "iter/sec",
            "range": "stddev: 0.00000664408008222",
            "extra": "mean: 56.361311364 usec\nrounds: 11334"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 14771.613265,
            "unit": "iter/sec",
            "range": "stddev: 0.00000196737468153",
            "extra": "mean: 67.6974127373 usec\nrounds: 8463"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 22489.0628398,
            "unit": "iter/sec",
            "range": "stddev: 0.00000145294969771",
            "extra": "mean: 44.4660592183 usec\nrounds: 17883"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_wide_speed",
            "value": 47389.8098595,
            "unit": "iter/sec",
            "range": "stddev: 9.23075630468e-7",
            "extra": "mean: 21.1015828712 usec\nrounds: 26330"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 3061.19653145,
            "unit": "iter/sec",
            "range": "stddev: 0.00003289187357",
            "extra": "mean: 326.66965016 usec\nrounds: 1252"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 1053329.62395,
            "unit": "iter/sec",
            "range": "stddev: 0.00000141383362799",
            "extra": "mean: 949.37043188 nsec\nrounds: 170941"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_device[cpu]",
            "value": 881368.302555,
            "unit": "iter/sec",
            "range": "stddev: 2.4121949216e-7",
            "extra": "mean: 1.13459945984 usec\nrounds: 106259"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_device[device1]",
            "value": 794013.221269,
            "unit": "iter/sec",
            "range": "stddev: 2.16611583405e-7",
            "extra": "mean: 1.25942487255 usec\nrounds: 82919"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 414116.699178,
            "unit": "iter/sec",
            "range": "stddev: 2.88372315604e-7",
            "extra": "mean: 2.4147782545 usec\nrounds: 59663"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 192605.251411,
            "unit": "iter/sec",
            "range": "stddev: 4.28619895182e-7",
            "extra": "mean: 5.19196643225 usec\nrounds: 33693"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 164597.33744,
            "unit": "iter/sec",
            "range": "stddev: 5.22177269394e-7",
            "extra": "mean: 6.07543241921 usec\nrounds: 30704"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_many_keys[10]",
            "value": 166778.278003,
            "unit": "iter/sec",
            "range": "stddev: 4.79832274834e-7",
            "extra": "mean: 5.99598468083 usec\nrounds: 59207"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_many_keys[50]",
            "value": 41673.6149044,
            "unit": "iter/sec",
            "range": "stddev: 9.62188149935e-7",
            "extra": "mean: 23.9959984823 usec\nrounds: 28994"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_many_keys[100]",
            "value": 21522.8650108,
            "unit": "iter/sec",
            "range": "stddev: 0.00000141810360271",
            "extra": "mean: 46.4622158573 usec\nrounds: 17317"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_many_keys[10]",
            "value": 69881.3113789,
            "unit": "iter/sec",
            "range": "stddev: 7.7421606699e-7",
            "extra": "mean: 14.309977593 usec\nrounds: 23787"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_many_keys[50]",
            "value": 19644.3175909,
            "unit": "iter/sec",
            "range": "stddev: 0.00000144396598972",
            "extra": "mean: 50.9053060955 usec\nrounds: 13731"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_from_namedtuple[namedtuple]",
            "value": 232214.907838,
            "unit": "iter/sec",
            "range": "stddev: 4.07783707834e-7",
            "extra": "mean: 4.30635573448 usec\nrounds: 31546"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_from_namedtuple[structseq]",
            "value": 221188.724228,
            "unit": "iter/sec",
            "range": "stddev: 5.12621397851e-7",
            "extra": "mean: 4.52102612143 usec\nrounds: 30741"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lazy_stack_flags",
            "value": 893688.903148,
            "unit": "iter/sec",
            "range": "stddev: 3.52485148113e-7",
            "extra": "mean: 1.11895761095 usec\nrounds: 80516"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 155144.91224,
            "unit": "iter/sec",
            "range": "stddev: 5.80311834868e-7",
            "extra": "mean: 6.44558681017 usec\nrounds: 16467"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone_device",
            "value": 56692.0045214,
            "unit": "iter/sec",
            "range": "stddev: 0.0000460029375705",
            "extra": "mean: 17.6391716688 usec\nrounds: 6297"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 114259.814969,
            "unit": "iter/sec",
            "range": "stddev: 0.0000319429648452",
            "extra": "mean: 8.75198336589 usec\nrounds: 14368"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice]",
            "value": 90015.2258594,
            "unit": "iter/sec",
            "range": "stddev: 0.00000633667944268",
            "extra": "mean: 11.1092316933 usec\nrounds: 16060"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 73111.0524617,
            "unit": "iter/sec",
            "range": "stddev: 0.00000614036831272",
            "extra": "mean: 13.6778225224 usec\nrounds: 15540"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 46101.7105725,
            "unit": "iter/sec",
            "range": "stddev: 0.00000703131677239",
            "extra": "mean: 21.6911691037 usec\nrounds: 5813"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 80792.4944527,
            "unit": "iter/sec",
            "range": "stddev: 0.00000628633870069",
            "extra": "mean: 12.3773873647 usec\nrounds: 13961"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 48409.6067374,
            "unit": "iter/sec",
            "range": "stddev: 0.00000700395898053",
            "extra": "mean: 20.6570568818 usec\nrounds: 8421"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tensor]",
            "value": 62696.4704584,
            "unit": "iter/sec",
            "range": "stddev: 0.00000675019677519",
            "extra": "mean: 15.9498611755 usec\nrounds: 16827"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[mask]",
            "value": 40907.2271735,
            "unit": "iter/sec",
            "range": "stddev: 0.00000730502386886",
            "extra": "mean: 24.4455581347 usec\nrounds: 4911"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[ellipsis]",
            "value": 64117.5039099,
            "unit": "iter/sec",
            "range": "stddev: 0.00000629734678148",
            "extra": "mean: 15.5963650955 usec\nrounds: 14514"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[none]",
            "value": 68984.7139801,
            "unit": "iter/sec",
            "range": "stddev: 0.00000661830732647",
            "extra": "mean: 14.4959650088 usec\nrounds: 15261"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[separated_tensors]",
            "value": 33736.4005407,
            "unit": "iter/sec",
            "range": "stddev: 0.0000067734205169",
            "extra": "mean: 29.6415736111 usec\nrounds: 5237"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_mask_many_leaves[None]",
            "value": 18776.918123,
            "unit": "iter/sec",
            "range": "stddev: 0.00000861667770307",
            "extra": "mean: 53.2568759926 usec\nrounds: 6298"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_mask_many_leaves[cpu]",
            "value": 15612.1824954,
            "unit": "iter/sec",
            "range": "stddev: 0.00192507533851",
            "extra": "mean: 64.0525436016 usec\nrounds: 7408"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 85638.7498907,
            "unit": "iter/sec",
            "range": "stddev: 0.00000113260372936",
            "extra": "mean: 11.6769569999 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 52334.2597539,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010907921997",
            "extra": "mean: 19.1079420002 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 39486.9648204,
            "unit": "iter/sec",
            "range": "stddev: 0.00000158445412081",
            "extra": "mean: 25.3248129997 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 64475.2226207,
            "unit": "iter/sec",
            "range": "stddev: 0.00000102037167571",
            "extra": "mean: 15.5098340006 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_non_tensor_rows",
            "value": 7351.65733943,
            "unit": "iter/sec",
            "range": "stddev: 0.00000865757980664",
            "extra": "mean: 136.023749997 usec\nrounds: 20"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cpu-None]",
            "value": 23110.5417663,
            "unit": "iter/sec",
            "range": "stddev: 0.00000863495689",
            "extra": "mean: 43.2702967377 usec\nrounds: 9136"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cpu-cpu]",
            "value": 24087.9970072,
            "unit": "iter/sec",
            "range": "stddev: 0.00000941805716674",
            "extra": "mean: 41.5144521855 usec\nrounds: 11942"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_non_tensor_rows[tensor]",
            "value": 6770.66129442,
            "unit": "iter/sec",
            "range": "stddev: 0.00000664299382083",
            "extra": "mean: 147.696060476 usec\nrounds: 3026"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_non_tensor_rows[mask]",
            "value": 6430.87520341,
            "unit": "iter/sec",
            "range": "stddev: 0.00000389715640425",
            "extra": "mean: 155.499829863 usec\nrounds: 3938"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 125501.512073,
            "unit": "iter/sec",
            "range": "stddev: 5.94264054018e-7",
            "extra": "mean: 7.96803148811 usec\nrounds: 18229"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 126294.865423,
            "unit": "iter/sec",
            "range": "stddev: 6.12507658905e-7",
            "extra": "mean: 7.91797826977 usec\nrounds: 27013"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 8503.26019451,
            "unit": "iter/sec",
            "range": "stddev: 0.0000423948050396",
            "extra": "mean: 117.601952325 usec\nrounds: 2559"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 145370.909499,
            "unit": "iter/sec",
            "range": "stddev: 0.0000078978597192",
            "extra": "mean: 6.87895538004 usec\nrounds: 52824"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 92340.1257221,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020715484546",
            "extra": "mean: 10.8295282487 usec\nrounds: 43347"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 91251.7374552,
            "unit": "iter/sec",
            "range": "stddev: 0.00000128312161985",
            "extra": "mean: 10.9586954494 usec\nrounds: 15866"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_many_leaves[False-False-16]",
            "value": 76818.2844559,
            "unit": "iter/sec",
            "range": "stddev: 7.23410004008e-7",
            "extra": "mean: 13.0177340861 usec\nrounds: 35064"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_many_leaves[False-False-256]",
            "value": 5977.63176784,
            "unit": "iter/sec",
            "range": "stddev: 0.00000307393291421",
            "extra": "mean: 167.290331495 usec\nrounds: 5077"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_many_leaves[False-False-1024]",
            "value": 1527.88563595,
            "unit": "iter/sec",
            "range": "stddev: 0.00000742920949588",
            "extra": "mean: 654.499248158 usec\nrounds: 1358"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_many_leaves[False-True-16]",
            "value": 23367.9435172,
            "unit": "iter/sec",
            "range": "stddev: 0.00000151376553822",
            "extra": "mean: 42.7936672847 usec\nrounds: 9170"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_many_leaves[False-True-256]",
            "value": 1893.86486114,
            "unit": "iter/sec",
            "range": "stddev: 0.00000596717346063",
            "extra": "mean: 528.020779369 usec\nrounds: 1396"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_many_leaves[False-True-1024]",
            "value": 480.013320457,
            "unit": "iter/sec",
            "range": "stddev: 0.0000198005979268",
            "extra": "mean: 2.08327552046 msec\nrounds: 440"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_many_leaves[True-False-16]",
            "value": 75599.5039163,
            "unit": "iter/sec",
            "range": "stddev: 7.79624022986e-7",
            "extra": "mean: 13.2276000264 usec\nrounds: 37580"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_many_leaves[True-False-256]",
            "value": 5994.8767776,
            "unit": "iter/sec",
            "range": "stddev: 0.00000303683198246",
            "extra": "mean: 166.80910002 usec\nrounds: 5139"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_many_leaves[True-False-1024]",
            "value": 1523.9861762,
            "unit": "iter/sec",
            "range": "stddev: 0.00000731782248985",
            "extra": "mean: 656.173930984 usec\nrounds: 1420"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_many_leaves[True-True-16]",
            "value": 19522.9267305,
            "unit": "iter/sec",
            "range": "stddev: 0.00000180273410738",
            "extra": "mean: 51.2218282536 usec\nrounds: 8728"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_many_leaves[True-True-256]",
            "value": 1858.3225712,
            "unit": "iter/sec",
            "range": "stddev: 0.00000602863885817",
            "extra": "mean: 538.119708333 usec\nrounds: 1320"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_many_leaves[True-True-1024]",
            "value": 478.854880195,
            "unit": "iter/sec",
            "range": "stddev: 0.0000253210710584",
            "extra": "mean: 2.08831535682 msec\nrounds: 440"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__keys_to_update[tensors]",
            "value": 100798.951664,
            "unit": "iter/sec",
            "range": "stddev: 6.41320746718e-7",
            "extra": "mean: 9.9207380979 usec\nrounds: 28209"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__keys_to_update[nested]",
            "value": 76577.6876612,
            "unit": "iter/sec",
            "range": "stddev: 0.00000110384255952",
            "extra": "mean: 13.0586340557 usec\nrounds: 25012"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__keys_to_update[non-tensor]",
            "value": 54710.4944496,
            "unit": "iter/sec",
            "range": "stddev: 9.6035550742e-7",
            "extra": "mean: 18.2780289241 usec\nrounds: 14659"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_copy__row_flat[64]",
            "value": 92047.9554745,
            "unit": "iter/sec",
            "range": "stddev: 0.00000597333540809",
            "extra": "mean: 10.8639023523 usec\nrounds: 12842"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_copy__row_flat[4096]",
            "value": 27729.2293689,
            "unit": "iter/sec",
            "range": "stddev: 0.00000599904178703",
            "extra": "mean: 36.0630288962 usec\nrounds: 5537"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_tensor_leaves[False-dict-16]",
            "value": 118307.87556,
            "unit": "iter/sec",
            "range": "stddev: 6.30641061816e-7",
            "extra": "mean: 8.45252266822 usec\nrounds: 59489"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_tensor_leaves[False-dict-256]",
            "value": 8310.34850664,
            "unit": "iter/sec",
            "range": "stddev: 0.00000248162486786",
            "extra": "mean: 120.331896936 usec\nrounds: 6200"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_tensor_leaves[False-td-16]",
            "value": 111882.218796,
            "unit": "iter/sec",
            "range": "stddev: 5.94636998222e-7",
            "extra": "mean: 8.93797075858 usec\nrounds: 62035"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_tensor_leaves[False-td-256]",
            "value": 8190.20660856,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023764968758",
            "extra": "mean: 122.097041967 usec\nrounds: 5981"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_tensor_leaves[False-kwargs-16]",
            "value": 95973.0876559,
            "unit": "iter/sec",
            "range": "stddev: 7.10771607582e-7",
            "extra": "mean: 10.4195876618 usec\nrounds: 53735"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_tensor_leaves[False-kwargs-256]",
            "value": 6811.89592119,
            "unit": "iter/sec",
            "range": "stddev: 0.00000270316557815",
            "extra": "mean: 146.802008071 usec\nrounds: 5080"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_tensor_leaves[True-dict-16]",
            "value": 83016.9436103,
            "unit": "iter/sec",
            "range": "stddev: 6.72741180998e-7",
            "extra": "mean: 12.0457337564 usec\nrounds: 46018"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_tensor_leaves[True-dict-256]",
            "value": 5630.42873423,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031212782123",
            "extra": "mean: 177.606368396 usec\nrounds: 4202"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_tensor_leaves[True-td-16]",
            "value": 69143.1754117,
            "unit": "iter/sec",
            "range": "stddev: 7.4926148904e-7",
            "extra": "mean: 14.4627433445 usec\nrounds: 27383"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_tensor_leaves[True-td-256]",
            "value": 4785.93076375,
            "unit": "iter/sec",
            "range": "stddev: 0.00000341806614132",
            "extra": "mean: 208.945772382 usec\nrounds: 3686"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_tensor_leaves[True-kwargs-16]",
            "value": 78192.270875,
            "unit": "iter/sec",
            "range": "stddev: 7.83333206422e-7",
            "extra": "mean: 12.7889878221 usec\nrounds: 43686"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_tensor_leaves[True-kwargs-256]",
            "value": 5350.13490912,
            "unit": "iter/sec",
            "range": "stddev: 0.00000316207776129",
            "extra": "mean: 186.911174575 usec\nrounds: 4067"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf[False-None-batch_size0]",
            "value": 1860709.87149,
            "unit": "iter/sec",
            "range": "stddev: 4.56838126461e-8",
            "extra": "mean: 537.429297992 nsec\nrounds: 158731"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf[False-None-batch_size1]",
            "value": 977204.552404,
            "unit": "iter/sec",
            "range": "stddev: 9.13758925333e-8",
            "extra": "mean: 1.0233272016 usec\nrounds: 193051"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf[False-cpu-batch_size0]",
            "value": 1510025.30497,
            "unit": "iter/sec",
            "range": "stddev: 5.24041944435e-8",
            "extra": "mean: 662.24055763 nsec\nrounds: 136613"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf[False-cpu-batch_size1]",
            "value": 886369.699497,
            "unit": "iter/sec",
            "range": "stddev: 9.7573685013e-8",
            "extra": "mean: 1.12819741082 usec\nrounds: 170941"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf[True-None-batch_size0]",
            "value": 931999.438314,
            "unit": "iter/sec",
            "range": "stddev: 9.98950148237e-8",
            "extra": "mean: 1.07296202003 usec\nrounds: 167225"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf[True-None-batch_size1]",
            "value": 630984.480835,
            "unit": "iter/sec",
            "range": "stddev: 1.59918556611e-7",
            "extra": "mean: 1.58482503195 usec\nrounds: 199204"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf[True-cpu-batch_size0]",
            "value": 813840.27039,
            "unit": "iter/sec",
            "range": "stddev: 1.17483116272e-7",
            "extra": "mean: 1.22874234218 usec\nrounds: 192679"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf[True-cpu-batch_size1]",
            "value": 591936.200838,
            "unit": "iter/sec",
            "range": "stddev: 1.56464934442e-7",
            "extra": "mean: 1.68937125079 usec\nrounds: 182150"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf_inplace[False-None-batch_size0]",
            "value": 556375.326113,
            "unit": "iter/sec",
            "range": "stddev: 2.63707856862e-7",
            "extra": "mean: 1.79734785686 usec\nrounds: 33922"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf_inplace[False-None-batch_size1]",
            "value": 435788.891347,
            "unit": "iter/sec",
            "range": "stddev: 3.31560884546e-7",
            "extra": "mean: 2.29468905669 usec\nrounds: 76924"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf_inplace[False-cpu-batch_size0]",
            "value": 517176.228132,
            "unit": "iter/sec",
            "range": "stddev: 2.61406044272e-7",
            "extra": "mean: 1.93357688462 usec\nrounds: 87033"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf_inplace[False-cpu-batch_size1]",
            "value": 415052.729336,
            "unit": "iter/sec",
            "range": "stddev: 2.77651401218e-7",
            "extra": "mean: 2.40933242771 usec\nrounds: 83543"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf_inplace[True-None-batch_size0]",
            "value": 418774.184848,
            "unit": "iter/sec",
            "range": "stddev: 3.13160051725e-7",
            "extra": "mean: 2.3879217874 usec\nrounds: 86431"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf_inplace[True-None-batch_size1]",
            "value": 350187.450563,
            "unit": "iter/sec",
            "range": "stddev: 3.92741053211e-7",
            "extra": "mean: 2.85561346756 usec\nrounds: 73153"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf_inplace[True-cpu-batch_size0]",
            "value": 394475.808999,
            "unit": "iter/sec",
            "range": "stddev: 3.54367290328e-7",
            "extra": "mean: 2.53500969435 usec\nrounds: 86648"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_tensor_leaf_inplace[True-cpu-batch_size1]",
            "value": 336649.291688,
            "unit": "iter/sec",
            "range": "stddev: 3.34905441024e-7",
            "extra": "mean: 2.97045033122 usec\nrounds: 76688"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 120175.108069,
            "unit": "iter/sec",
            "range": "stddev: 0.00000122860179673",
            "extra": "mean: 8.32119076962 usec\nrounds: 24679"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 89995.9790166,
            "unit": "iter/sec",
            "range": "stddev: 0.00000175112035493",
            "extra": "mean: 11.111607551 usec\nrounds: 24686"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unravel_key[str]",
            "value": 12675906.4858,
            "unit": "iter/sec",
            "range": "stddev: 5.51673430546e-9",
            "extra": "mean: 78.8898214988 nsec\nrounds: 123305"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unravel_key[tuple]",
            "value": 4089003.26352,
            "unit": "iter/sec",
            "range": "stddev: 2.20604019237e-8",
            "extra": "mean: 244.55837659 nsec\nrounds: 185529"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unravel_key[nested_tuple]",
            "value": 1569361.29315,
            "unit": "iter/sec",
            "range": "stddev: 1.91326832923e-7",
            "extra": "mean: 637.201901415 nsec\nrounds: 140846"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unravel_key_list[str]",
            "value": 3744301.25847,
            "unit": "iter/sec",
            "range": "stddev: 2.38632687309e-8",
            "extra": "mean: 267.07252728 nsec\nrounds: 176648"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unravel_key_list[tuple]",
            "value": 1340865.08899,
            "unit": "iter/sec",
            "range": "stddev: 1.98324943301e-7",
            "extra": "mean: 745.787184863 nsec\nrounds: 162075"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unravel_key_list[mixed]",
            "value": 792722.501421,
            "unit": "iter/sec",
            "range": "stddev: 4.17676364156e-7",
            "extra": "mean: 1.26147548254 usec\nrounds: 147711"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unravel_key_to_tuple[str]",
            "value": 9611597.767,
            "unit": "iter/sec",
            "range": "stddev: 6.76114943732e-9",
            "extra": "mean: 104.040974689 nsec\nrounds: 91660"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unravel_key_to_tuple[tuple]",
            "value": 4232775.31694,
            "unit": "iter/sec",
            "range": "stddev: 2.15737997178e-8",
            "extra": "mean: 236.251613923 nsec\nrounds: 199204"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unravel_key_to_tuple[nested_tuple]",
            "value": 1796341.75777,
            "unit": "iter/sec",
            "range": "stddev: 5.20078856013e-8",
            "extra": "mean: 556.686942045 nsec\nrounds: 182482"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_get_by_key[str]",
            "value": 1834846.66671,
            "unit": "iter/sec",
            "range": "stddev: 3.558536072e-8",
            "extra": "mean: 545.00466886 nsec\nrounds: 87109"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_get_by_key[tuple]",
            "value": 1042083.32705,
            "unit": "iter/sec",
            "range": "stddev: 9.30472045619e-8",
            "extra": "mean: 959.616159321 nsec\nrounds: 198808"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_get_by_key[nested_tuple]",
            "value": 718907.274333,
            "unit": "iter/sec",
            "range": "stddev: 2.59337088278e-7",
            "extra": "mean: 1.39099997413 usec\nrounds: 193051"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_by_key[x]",
            "value": 5213236.81849,
            "unit": "iter/sec",
            "range": "stddev: 9.3592928205e-9",
            "extra": "mean: 191.819407945 nsec\nrounds: 50350"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_by_key[sub]",
            "value": 4891806.26356,
            "unit": "iter/sec",
            "range": "stddev: 1.87329764558e-8",
            "extra": "mean: 204.423467759 nsec\nrounds: 194932"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_by_key[text]",
            "value": 597652.322556,
            "unit": "iter/sec",
            "range": "stddev: 2.80792056605e-7",
            "extra": "mean: 1.67321360975 usec\nrounds: 49876"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_by_key[key3]",
            "value": 1106994.26827,
            "unit": "iter/sec",
            "range": "stddev: 2.08473368972e-7",
            "extra": "mean: 903.347044034 nsec\nrounds: 183824"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_get_nested_with_default[hit]",
            "value": 1014253.46446,
            "unit": "iter/sec",
            "range": "stddev: 1.88409958689e-7",
            "extra": "mean: 985.946841733 nsec\nrounds: 193799"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_get_nested_with_default[miss]",
            "value": 971568.650633,
            "unit": "iter/sec",
            "range": "stddev: 2.02611368801e-7",
            "extra": "mean: 1.02926334577 usec\nrounds: 160257"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 52441.3338308,
            "unit": "iter/sec",
            "range": "stddev: 0.00000120780629618",
            "extra": "mean: 19.0689276369 usec\nrounds: 8941"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 22262.3098511,
            "unit": "iter/sec",
            "range": "stddev: 0.00000158552030494",
            "extra": "mean: 44.9189687274 usec\nrounds: 13782"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 22259.4477461,
            "unit": "iter/sec",
            "range": "stddev: 0.00000186402625265",
            "extra": "mean: 44.9247443786 usec\nrounds: 13786"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_select_update_step[exclude]",
            "value": 167597.824312,
            "unit": "iter/sec",
            "range": "stddev: 5.15197128293e-7",
            "extra": "mean: 5.96666456802 usec\nrounds: 55433"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_select_update_step[select]",
            "value": 153433.434147,
            "unit": "iter/sec",
            "range": "stddev: 6.04360179238e-7",
            "extra": "mean: 6.51748431207 usec\nrounds: 51760"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_select_update_step[step]",
            "value": 61168.2633206,
            "unit": "iter/sec",
            "range": "stddev: 9.82900295107e-7",
            "extra": "mean: 16.3483470956 usec\nrounds: 24051"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 10391.8817079,
            "unit": "iter/sec",
            "range": "stddev: 0.00000442000062229",
            "extra": "mean: 96.2289629642 usec\nrounds: 162"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 1234590.79892,
            "unit": "iter/sec",
            "range": "stddev: 4.57187743864e-8",
            "extra": "mean: 809.984977109 nsec\nrounds: 60278"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_same_device",
            "value": 92211.4746713,
            "unit": "iter/sec",
            "range": "stddev: 6.51322861251e-7",
            "extra": "mean: 10.8446373248 usec\nrounds: 14280"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_dtype_nested",
            "value": 61635.4405115,
            "unit": "iter/sec",
            "range": "stddev: 9.53173177444e-7",
            "extra": "mean: 16.2244317831 usec\nrounds: 5248"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 5078.9144685,
            "unit": "iter/sec",
            "range": "stddev: 0.00000369097034831",
            "extra": "mean: 196.892467121 usec\nrounds: 1323"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 5121.94635245,
            "unit": "iter/sec",
            "range": "stddev: 0.00000366568631739",
            "extra": "mean: 195.23828076 usec\nrounds: 4470"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1673.91224383,
            "unit": "iter/sec",
            "range": "stddev: 0.00618592366194",
            "extra": "mean: 597.402882789 usec\nrounds: 2022"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stack_nested",
            "value": 3249.37748011,
            "unit": "iter/sec",
            "range": "stddev: 0.0000267580982469",
            "extra": "mean: 307.751255778 usec\nrounds: 1947"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stack_rollout_steps",
            "value": 10538.2856392,
            "unit": "iter/sec",
            "range": "stddev: 0.00000756067431502",
            "extra": "mean: 94.8920948094 usec\nrounds: 5221"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_cat_nested",
            "value": 3730.33676543,
            "unit": "iter/sec",
            "range": "stddev: 0.00000461470538275",
            "extra": "mean: 268.072311665 usec\nrounds: 2323"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_pack_nested[pack]",
            "value": 39297.8820826,
            "unit": "iter/sec",
            "range": "stddev: 0.00000152993039667",
            "extra": "mean: 25.4466639677 usec\nrounds: 19513"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_pack_nested[unpack]",
            "value": 48634.2579692,
            "unit": "iter/sec",
            "range": "stddev: 0.00000106937821043",
            "extra": "mean: 20.561637861 usec\nrounds: 9593"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_apply_nested",
            "value": 8977.93176416,
            "unit": "iter/sec",
            "range": "stddev: 0.00000244624407091",
            "extra": "mean: 111.384228157 usec\nrounds: 4624"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_pad_nested[True]",
            "value": 23287.0852015,
            "unit": "iter/sec",
            "range": "stddev: 0.00000166761754928",
            "extra": "mean: 42.9422571072 usec\nrounds: 4010"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_pad_nested[False]",
            "value": 25037.1066791,
            "unit": "iter/sec",
            "range": "stddev: 0.00000145676896059",
            "extra": "mean: 39.9407173047 usec\nrounds: 11072"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_pointwise_nested[neg]",
            "value": 82430.0669216,
            "unit": "iter/sec",
            "range": "stddev: 0.00000497945428988",
            "extra": "mean: 12.1314956708 usec\nrounds: 9240"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_pointwise_nested[mul_scalar]",
            "value": 52106.8246698,
            "unit": "iter/sec",
            "range": "stddev: 0.00000115205821214",
            "extra": "mean: 19.1913440579 usec\nrounds: 4258"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_pointwise_nested[add]",
            "value": 48958.0913586,
            "unit": "iter/sec",
            "range": "stddev: 0.00000369070024053",
            "extra": "mean: 20.4256328678 usec\nrounds: 11105"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_shape_ops_nested[reshape]",
            "value": 50452.9220113,
            "unit": "iter/sec",
            "range": "stddev: 0.000001122561602",
            "extra": "mean: 19.8204575699 usec\nrounds: 7695"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_shape_ops_nested[view_flatten]",
            "value": 22358.5831988,
            "unit": "iter/sec",
            "range": "stddev: 0.00000184508928235",
            "extra": "mean: 44.7255530955 usec\nrounds: 5556"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_shape_ops_nested[unsqueeze]",
            "value": 59533.2701405,
            "unit": "iter/sec",
            "range": "stddev: 9.68366929984e-7",
            "extra": "mean: 16.7973302599 usec\nrounds: 12905"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_shape_ops_nested[unbind]",
            "value": 24439.6128602,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031998055245",
            "extra": "mean: 40.9171784235 usec\nrounds: 7953"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_shape_ops_nested[clone]",
            "value": 131896.831225,
            "unit": "iter/sec",
            "range": "stddev: 0.00000167415417881",
            "extra": "mean: 7.58168328013 usec\nrounds: 24498"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_shape_ops_nested[eq_all]",
            "value": 37552.2465509,
            "unit": "iter/sec",
            "range": "stddev: 0.00000127284679318",
            "extra": "mean: 26.6295652551 usec\nrounds: 6873"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 1395.83715565,
            "unit": "iter/sec",
            "range": "stddev: 0.0000160509438119",
            "extra": "mean: 716.415948633 usec\nrounds: 915"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 1432.37129262,
            "unit": "iter/sec",
            "range": "stddev: 0.00000717594583476",
            "extra": "mean: 698.143006045 usec\nrounds: 1158"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate_njt[False-None]",
            "value": 358.932299266,
            "unit": "iter/sec",
            "range": "stddev: 0.0000250278605088",
            "extra": "mean: 2.78604071588 msec\nrounds: 359"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 16013.803638,
            "unit": "iter/sec",
            "range": "stddev: 0.000028242517387",
            "extra": "mean: 62.4461260175 usec\nrounds: 3071"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 14847.6253803,
            "unit": "iter/sec",
            "range": "stddev: 0.0000215342915796",
            "extra": "mean: 67.3508372137 usec\nrounds: 5541"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 303560.408502,
            "unit": "iter/sec",
            "range": "stddev: 0.00000767420769503",
            "extra": "mean: 3.2942372325 usec\nrounds: 23458"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 3776913.51501,
            "unit": "iter/sec",
            "range": "stddev: 1.11752672889e-7",
            "extra": "mean: 264.766454415 nsec\nrounds: 166390"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 461492.30497,
            "unit": "iter/sec",
            "range": "stddev: 8.04927829488e-7",
            "extra": "mean: 2.16688336778 usec\nrounds: 36285"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 6640.58853434,
            "unit": "iter/sec",
            "range": "stddev: 0.0000263907679043",
            "extra": "mean: 150.589062224 usec\nrounds: 2491"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 5283.08685602,
            "unit": "iter/sec",
            "range": "stddev: 0.00000838217725935",
            "extra": "mean: 189.283278366 usec\nrounds: 3305"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 3433.12562832,
            "unit": "iter/sec",
            "range": "stddev: 0.0000238080774667",
            "extra": "mean: 291.279757359 usec\nrounds: 2514"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 19.9439500016,
            "unit": "iter/sec",
            "range": "stddev: 0.00107363354725",
            "extra": "mean: 50.1405188 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_archive",
            "value": 6.92638149009,
            "unit": "iter/sec",
            "range": "stddev: 0.000861475115277",
            "extra": "mean: 144.3755302 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_load_memmap",
            "value": 59.8181139852,
            "unit": "iter/sec",
            "range": "stddev: 0.000189928597887",
            "extra": "mean: 16.7173441852 msec\nrounds: 54"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_load_archive",
            "value": 98.3014348874,
            "unit": "iter/sec",
            "range": "stddev: 0.0181900584675",
            "extra": "mean: 10.1727914872 msec\nrounds: 117"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_zarr",
            "value": 0.727807966355,
            "unit": "iter/sec",
            "range": "stddev: 0.0430293786125",
            "extra": "mean: 1.3739888078 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_load_zarr",
            "value": 5.46692221357,
            "unit": "iter/sec",
            "range": "stddev: 0.00556615859928",
            "extra": "mean: 182.9182785 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 38.9385816145,
            "unit": "iter/sec",
            "range": "stddev: 0.00154317291875",
            "extra": "mean: 25.6814696 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 19.0101297682,
            "unit": "iter/sec",
            "range": "stddev: 0.00088674482079",
            "extra": "mean: 52.6035336 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 10.664456286,
            "unit": "iter/sec",
            "range": "stddev: 0.0886302672245",
            "extra": "mean: 93.7694312 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 38.4908762074,
            "unit": "iter/sec",
            "range": "stddev: 0.00230542881602",
            "extra": "mean: 25.9801828 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_filesystem",
            "value": 28.7340422336,
            "unit": "iter/sec",
            "range": "stddev: 0.000309554176235",
            "extra": "mean: 34.8019256 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_filesystem",
            "value": 28.709307399,
            "unit": "iter/sec",
            "range": "stddev: 0.000343465803728",
            "extra": "mean: 34.8319096 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_consolidate[few_large-0]",
            "value": 344.21045857,
            "unit": "iter/sec",
            "range": "stddev: 0.00568839062582",
            "extra": "mean: 2.90519934855 msec\nrounds: 482"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_consolidate[few_large-8]",
            "value": 339.083683988,
            "unit": "iter/sec",
            "range": "stddev: 0.00534973791881",
            "extra": "mean: 2.9491245 msec\nrounds: 86"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_consolidate[many_small-0]",
            "value": 15.74039107,
            "unit": "iter/sec",
            "range": "stddev: 0.00116349290765",
            "extra": "mean: 63.5308230624 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_consolidate[many_small-8]",
            "value": 18.4441116055,
            "unit": "iter/sec",
            "range": "stddev: 0.0437670354744",
            "extra": "mean: 54.2178458571 msec\nrounds: 21"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_consolidate_to_file[few_large-0]",
            "value": 166.314484658,
            "unit": "iter/sec",
            "range": "stddev: 0.0000861064723001",
            "extra": "mean: 6.01270540001 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_consolidate_to_file[few_large-8]",
            "value": 162.911279299,
            "unit": "iter/sec",
            "range": "stddev: 0.000113247143889",
            "extra": "mean: 6.13831040002 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_consolidate_to_file[many_small-0]",
            "value": 37.7922642077,
            "unit": "iter/sec",
            "range": "stddev: 0.000738566069804",
            "extra": "mean: 26.4604416 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_consolidate_to_file[many_small-8]",
            "value": 37.8304030798,
            "unit": "iter/sec",
            "range": "stddev: 0.00126472016672",
            "extra": "mean: 26.4337654 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 59293.0433578,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113668127469",
            "extra": "mean: 16.8653849317 usec\nrounds: 13870"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 38687.1872068,
            "unit": "iter/sec",
            "range": "stddev: 0.00000160493866002",
            "extra": "mean: 25.8483511519 usec\nrounds: 6903"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 58549.3293401,
            "unit": "iter/sec",
            "range": "stddev: 0.00000708131157615",
            "extra": "mean: 17.0796149379 usec\nrounds: 16535"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 33892.9178993,
            "unit": "iter/sec",
            "range": "stddev: 0.00000162164645116",
            "extra": "mean: 29.5046889433 usec\nrounds: 10220"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 52835.2841133,
            "unit": "iter/sec",
            "range": "stddev: 0.00000720997100244",
            "extra": "mean: 18.9267459574 usec\nrounds: 12120"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 34820.1710511,
            "unit": "iter/sec",
            "range": "stddev: 0.00000153622333711",
            "extra": "mean: 28.7189858583 usec\nrounds: 12092"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 43555.1234026,
            "unit": "iter/sec",
            "range": "stddev: 0.0000072156546446",
            "extra": "mean: 22.959411474 usec\nrounds: 11748"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 26239.8057528,
            "unit": "iter/sec",
            "range": "stddev: 0.00000187463469484",
            "extra": "mean: 38.1100382153 usec\nrounds: 9682"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 46393.8691798,
            "unit": "iter/sec",
            "range": "stddev: 0.00000711498605469",
            "extra": "mean: 21.5545721381 usec\nrounds: 9662"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 38335.0758493,
            "unit": "iter/sec",
            "range": "stddev: 0.00000107208665773",
            "extra": "mean: 26.0857707425 usec\nrounds: 11040"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-compile]",
            "value": 23023.5578902,
            "unit": "iter/sec",
            "range": "stddev: 0.00000678734766065",
            "extra": "mean: 43.4337735622 usec\nrounds: 12308"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-eager]",
            "value": 14497.777361,
            "unit": "iter/sec",
            "range": "stddev: 0.00001631814726",
            "extra": "mean: 68.9760902725 usec\nrounds: 11709"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-compile]",
            "value": 46654.9885348,
            "unit": "iter/sec",
            "range": "stddev: 0.00000618239117038",
            "extra": "mean: 21.4339351783 usec\nrounds: 25948"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-eager]",
            "value": 14504.4384388,
            "unit": "iter/sec",
            "range": "stddev: 0.0000180863552615",
            "extra": "mean: 68.9444134095 usec\nrounds: 12767"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-compile]",
            "value": 42095.8168876,
            "unit": "iter/sec",
            "range": "stddev: 0.00000112396045686",
            "extra": "mean: 23.7553294825 usec\nrounds: 27786"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-eager]",
            "value": 47004.7653199,
            "unit": "iter/sec",
            "range": "stddev: 9.76389908018e-7",
            "extra": "mean: 21.2744387339 usec\nrounds: 39557"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-compile]",
            "value": 107237.754246,
            "unit": "iter/sec",
            "range": "stddev: 0.00000107166864083",
            "extra": "mean: 9.32507405657 usec\nrounds: 58928"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-eager]",
            "value": 25886.9500694,
            "unit": "iter/sec",
            "range": "stddev: 0.00000851200916598",
            "extra": "mean: 38.6295024064 usec\nrounds: 22858"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-compile]",
            "value": 18037.464078,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022641772599",
            "extra": "mean: 55.4401658502 usec\nrounds: 11022"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-eager]",
            "value": 11643.4639931,
            "unit": "iter/sec",
            "range": "stddev: 0.00000284655791329",
            "extra": "mean: 85.8850940402 usec\nrounds: 10453"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-compile]",
            "value": 38996.6151248,
            "unit": "iter/sec",
            "range": "stddev: 0.00000171149605581",
            "extra": "mean: 25.6432512617 usec\nrounds: 19418"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-eager]",
            "value": 35112.0414984,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113826195693",
            "extra": "mean: 28.4802579778 usec\nrounds: 24944"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-compile]",
            "value": 21587.077979,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023168162059",
            "extra": "mean: 46.3240092509 usec\nrounds: 15134"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-eager]",
            "value": 8430.63584585,
            "unit": "iter/sec",
            "range": "stddev: 0.00000955585902109",
            "extra": "mean: 118.615015318 usec\nrounds: 7899"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-eager]",
            "value": 10857.0442591,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029538759079",
            "extra": "mean: 92.1060996103 usec\nrounds: 9497"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-compile]",
            "value": 18128.5708518,
            "unit": "iter/sec",
            "range": "stddev: 0.00000228645379856",
            "extra": "mean: 55.1615462783 usec\nrounds: 12360"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-eager]",
            "value": 37840.4114773,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012501054257",
            "extra": "mean: 26.4267739424 usec\nrounds: 19787"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-compile]",
            "value": 37909.3351468,
            "unit": "iter/sec",
            "range": "stddev: 0.00000285059526039",
            "extra": "mean: 26.3787269317 usec\nrounds: 19724"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-eager]",
            "value": 10813.6708723,
            "unit": "iter/sec",
            "range": "stddev: 0.00000973175652364",
            "extra": "mean: 92.4755350712 usec\nrounds: 10051"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-compile]",
            "value": 21663.8838881,
            "unit": "iter/sec",
            "range": "stddev: 0.00000241190760336",
            "extra": "mean: 46.1597747276 usec\nrounds: 15226"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-compile]",
            "value": 65125.9770614,
            "unit": "iter/sec",
            "range": "stddev: 0.00000123280120141",
            "extra": "mean: 15.3548560056 usec\nrounds: 36050"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-eager]",
            "value": 92832.1511968,
            "unit": "iter/sec",
            "range": "stddev: 7.9313709885e-7",
            "extra": "mean: 10.7721299906 usec\nrounds: 68028"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-compile]",
            "value": 82809.9507521,
            "unit": "iter/sec",
            "range": "stddev: 0.00000374581281616",
            "extra": "mean: 12.0758434333 usec\nrounds: 47756"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-eager]",
            "value": 31062.5924923,
            "unit": "iter/sec",
            "range": "stddev: 0.00000865300136659",
            "extra": "mean: 32.193063095 usec\nrounds: 26056"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-compile]",
            "value": 6555.26044632,
            "unit": "iter/sec",
            "range": "stddev: 0.00000491891857994",
            "extra": "mean: 152.549240139 usec\nrounds: 3752"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-eager]",
            "value": 1093.44520557,
            "unit": "iter/sec",
            "range": "stddev: 0.0000119961266917",
            "extra": "mean: 914.540568571 usec\nrounds: 1050"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-compile]",
            "value": 7735.09597643,
            "unit": "iter/sec",
            "range": "stddev: 0.00000407313259424",
            "extra": "mean: 129.280878097 usec\nrounds: 5045"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-eager]",
            "value": 2267.12245307,
            "unit": "iter/sec",
            "range": "stddev: 0.0000155241470551",
            "extra": "mean: 441.087775671 usec\nrounds: 2162"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add_stack[compile]",
            "value": 3038.83946592,
            "unit": "iter/sec",
            "range": "stddev: 0.00000698851468649",
            "extra": "mean: 329.072993561 usec\nrounds: 2019"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add_stack[eager]",
            "value": 806.505413436,
            "unit": "iter/sec",
            "range": "stddev: 0.0000135467835173",
            "extra": "mean: 1.23991728182 msec\nrounds: 770"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-compile]",
            "value": 44202.193238,
            "unit": "iter/sec",
            "range": "stddev: 0.00000157347558782",
            "extra": "mean: 22.623311803 usec\nrounds: 23409"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-eager]",
            "value": 65117.5716298,
            "unit": "iter/sec",
            "range": "stddev: 0.00000733359394083",
            "extra": "mean: 15.3568380235 usec\nrounds: 45414"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-compile]",
            "value": 66491.023729,
            "unit": "iter/sec",
            "range": "stddev: 0.00000230253639588",
            "extra": "mean: 15.0396240563 usec\nrounds: 36955"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-eager]",
            "value": 71959.1983636,
            "unit": "iter/sec",
            "range": "stddev: 0.00000827052216481",
            "extra": "mean: 13.8967640377 usec\nrounds: 51949"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-compile]",
            "value": 63118.3186225,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012570561963",
            "extra": "mean: 15.8432610663 usec\nrounds: 40258"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-eager]",
            "value": 70523.8185495,
            "unit": "iter/sec",
            "range": "stddev: 0.0000085716046986",
            "extra": "mean: 14.179606558 usec\nrounds: 47148"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-compile]",
            "value": 33690.9287747,
            "unit": "iter/sec",
            "range": "stddev: 0.00000379448065656",
            "extra": "mean: 29.6815800683 usec\nrounds: 18403"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-eager]",
            "value": 87030.7946005,
            "unit": "iter/sec",
            "range": "stddev: 0.00000725802281366",
            "extra": "mean: 11.4901857968 usec\nrounds: 62267"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-compile]",
            "value": 46625.2762049,
            "unit": "iter/sec",
            "range": "stddev: 0.00000269542395142",
            "extra": "mean: 21.4475941248 usec\nrounds: 27028"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-eager]",
            "value": 85830.8869218,
            "unit": "iter/sec",
            "range": "stddev: 0.0000079543067154",
            "extra": "mean: 11.6508175071 usec\nrounds: 61997"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-compile]",
            "value": 45341.4470908,
            "unit": "iter/sec",
            "range": "stddev: 0.00000157220847823",
            "extra": "mean: 22.0548761489 usec\nrounds: 27848"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-eager]",
            "value": 86491.1099587,
            "unit": "iter/sec",
            "range": "stddev: 0.00000801078172705",
            "extra": "mean: 11.5618819145 usec\nrounds: 62929"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-compile]",
            "value": 33894.9196726,
            "unit": "iter/sec",
            "range": "stddev: 0.00000179143610919",
            "extra": "mean: 29.5029464492 usec\nrounds: 20952"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-eager]",
            "value": 88274.5709964,
            "unit": "iter/sec",
            "range": "stddev: 0.00000730675572133",
            "extra": "mean: 11.3282906811 usec\nrounds: 62732"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-compile]",
            "value": 45214.7287581,
            "unit": "iter/sec",
            "range": "stddev: 0.00000149454749426",
            "extra": "mean: 22.1166869174 usec\nrounds: 27028"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-eager]",
            "value": 86606.5354674,
            "unit": "iter/sec",
            "range": "stddev: 0.0000079630560964",
            "extra": "mean: 11.5464727298 usec\nrounds: 64851"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-compile]",
            "value": 45027.7961415,
            "unit": "iter/sec",
            "range": "stddev: 0.00000179329691898",
            "extra": "mean: 22.2085042061 usec\nrounds: 29482"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-eager]",
            "value": 85577.7552579,
            "unit": "iter/sec",
            "range": "stddev: 0.00000792091034077",
            "extra": "mean: 11.6852796265 usec\nrounds: 63735"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-eager]",
            "value": 84203.5183477,
            "unit": "iter/sec",
            "range": "stddev: 8.86550207652e-7",
            "extra": "mean: 11.8759883152 usec\nrounds: 45872"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-compile]",
            "value": 51171.7169073,
            "unit": "iter/sec",
            "range": "stddev: 0.00000141267206267",
            "extra": "mean: 19.5420451069 usec\nrounds: 25894"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-eager]",
            "value": 7782.40113879,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036575131665",
            "extra": "mean: 128.495046987 usec\nrounds: 7002"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-compile]",
            "value": 45732.191857,
            "unit": "iter/sec",
            "range": "stddev: 0.00000184634761213",
            "extra": "mean: 21.8664349858 usec\nrounds: 23518"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[eager]",
            "value": 34676.9407363,
            "unit": "iter/sec",
            "range": "stddev: 0.00000168759244897",
            "extra": "mean: 28.8376073196 usec\nrounds: 29291"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[compile]",
            "value": 51668.7772287,
            "unit": "iter/sec",
            "range": "stddev: 0.00000142041852609",
            "extra": "mean: 19.3540481048 usec\nrounds: 28043"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[eager]",
            "value": 29051.2613228,
            "unit": "iter/sec",
            "range": "stddev: 0.00000156852433957",
            "extra": "mean: 34.4219133514 usec\nrounds: 25563"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[compile]",
            "value": 37330.8473998,
            "unit": "iter/sec",
            "range": "stddev: 0.0000056935575186",
            "extra": "mean: 26.7874979984 usec\nrounds: 22233"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-eager]",
            "value": 179637.52974,
            "unit": "iter/sec",
            "range": "stddev: 5.7309905916e-7",
            "extra": "mean: 5.56676548295 usec\nrounds: 119761"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-compile]",
            "value": 81245.0061532,
            "unit": "iter/sec",
            "range": "stddev: 0.00000127741615674",
            "extra": "mean: 12.3084488186 usec\nrounds: 42320"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-eager]",
            "value": 110539.846014,
            "unit": "iter/sec",
            "range": "stddev: 7.96597716845e-7",
            "extra": "mean: 9.04651160695 usec\nrounds: 83613"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-compile]",
            "value": 68143.5430194,
            "unit": "iter/sec",
            "range": "stddev: 0.00000146635548088",
            "extra": "mean: 14.674904704 usec\nrounds: 36969"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-eager]",
            "value": 64018.0678419,
            "unit": "iter/sec",
            "range": "stddev: 7.98912178797e-7",
            "extra": "mean: 15.6205901508 usec\nrounds: 53161"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-compile]",
            "value": 49086.3237774,
            "unit": "iter/sec",
            "range": "stddev: 0.00000155279956323",
            "extra": "mean: 20.3722732331 usec\nrounds: 27848"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[eager]",
            "value": 64766.8553709,
            "unit": "iter/sec",
            "range": "stddev: 8.21951279213e-7",
            "extra": "mean: 15.4399961875 usec\nrounds: 53505"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[compile]",
            "value": 42688.9138064,
            "unit": "iter/sec",
            "range": "stddev: 0.00000144667274053",
            "extra": "mean: 23.4252856499 usec\nrounds: 24432"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_build_and_set_cold",
            "value": 4.63330633195,
            "unit": "iter/sec",
            "range": "stddev: 0.131603049837",
            "extra": "mean: 215.8285959 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[eager]",
            "value": 90057.7920371,
            "unit": "iter/sec",
            "range": "stddev: 8.74451852132e-7",
            "extra": "mean: 11.1039808703 usec\nrounds: 44015"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile]",
            "value": 34888.0399935,
            "unit": "iter/sec",
            "range": "stddev: 0.0000189564806",
            "extra": "mean: 28.6631177958 usec\nrounds: 9092"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile-overhead]",
            "value": 35681.9994558,
            "unit": "iter/sec",
            "range": "stddev: 0.0000128190589554",
            "extra": "mean: 28.0253353302 usec\nrounds: 9513"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[eager]",
            "value": 4476.78419405,
            "unit": "iter/sec",
            "range": "stddev: 0.0000365516276796",
            "extra": "mean: 223.37462711 usec\nrounds: 3969"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile]",
            "value": 3649.98599052,
            "unit": "iter/sec",
            "range": "stddev: 0.0000247848873893",
            "extra": "mean: 273.973654309 usec\nrounds: 2239"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile-overhead]",
            "value": 3651.23686056,
            "unit": "iter/sec",
            "range": "stddev: 0.0000300870690655",
            "extra": "mean: 273.879794215 usec\nrounds: 2420"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[eager]",
            "value": 181.652562307,
            "unit": "iter/sec",
            "range": "stddev: 0.000179854854244",
            "extra": "mean: 5.50501455801 msec\nrounds: 181"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile]",
            "value": 173.083509133,
            "unit": "iter/sec",
            "range": "stddev: 0.00032227525032",
            "extra": "mean: 5.77755792571 msec\nrounds: 175"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile-overhead]",
            "value": 168.905171318,
            "unit": "iter/sec",
            "range": "stddev: 0.000489629816338",
            "extra": "mean: 5.92048184314 msec\nrounds: 153"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[eager]",
            "value": 28172.3202078,
            "unit": "iter/sec",
            "range": "stddev: 0.00000147124099867",
            "extra": "mean: 35.495833947 usec\nrounds: 17928"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile]",
            "value": 28484.6168451,
            "unit": "iter/sec",
            "range": "stddev: 0.00000175732950831",
            "extra": "mean: 35.1066684674 usec\nrounds: 14991"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile-overhead]",
            "value": 29097.2518968,
            "unit": "iter/sec",
            "range": "stddev: 0.00000189824374347",
            "extra": "mean: 34.3675067166 usec\nrounds: 15409"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[eager]",
            "value": 3632.62828871,
            "unit": "iter/sec",
            "range": "stddev: 0.0000311167120232",
            "extra": "mean: 275.28277614 usec\nrounds: 3051"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile]",
            "value": 3465.69380085,
            "unit": "iter/sec",
            "range": "stddev: 0.0000278701388455",
            "extra": "mean: 288.542513408 usec\nrounds: 2051"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile-overhead]",
            "value": 3473.05661104,
            "unit": "iter/sec",
            "range": "stddev: 0.0000280337851963",
            "extra": "mean: 287.930809081 usec\nrounds: 2687"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-eager]",
            "value": 4624.40531596,
            "unit": "iter/sec",
            "range": "stddev: 0.0000133240806516",
            "extra": "mean: 216.244020944 usec\nrounds: 4154"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile]",
            "value": 4031.22795436,
            "unit": "iter/sec",
            "range": "stddev: 0.00000687991351846",
            "extra": "mean: 248.063372085 usec\nrounds: 3303"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile-overhead]",
            "value": 4118.84869581,
            "unit": "iter/sec",
            "range": "stddev: 0.00000718703611265",
            "extra": "mean: 242.786291474 usec\nrounds: 3249"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-eager]",
            "value": 2052.81434375,
            "unit": "iter/sec",
            "range": "stddev: 0.00000795418108984",
            "extra": "mean: 487.13611294 usec\nrounds: 1824"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile]",
            "value": 3771.72695406,
            "unit": "iter/sec",
            "range": "stddev: 0.00000686512018109",
            "extra": "mean: 265.130538923 usec\nrounds: 2993"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile-overhead]",
            "value": 3812.7493722,
            "unit": "iter/sec",
            "range": "stddev: 0.00000702987970236",
            "extra": "mean: 262.277926604 usec\nrounds: 3011"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-eager]",
            "value": 4612.65846816,
            "unit": "iter/sec",
            "range": "stddev: 0.0000125209912491",
            "extra": "mean: 216.79471977 usec\nrounds: 4193"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile]",
            "value": 4118.78784915,
            "unit": "iter/sec",
            "range": "stddev: 0.00000690341081827",
            "extra": "mean: 242.789878145 usec\nrounds: 3299"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile-overhead]",
            "value": 4220.97991588,
            "unit": "iter/sec",
            "range": "stddev: 0.00000629429447619",
            "extra": "mean: 236.911811932 usec\nrounds: 3302"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-eager]",
            "value": 1957.22179148,
            "unit": "iter/sec",
            "range": "stddev: 0.00000905109014288",
            "extra": "mean: 510.928298648 usec\nrounds: 1778"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile]",
            "value": 3085.16210878,
            "unit": "iter/sec",
            "range": "stddev: 0.00000752735582409",
            "extra": "mean: 324.132076286 usec\nrounds: 2543"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile-overhead]",
            "value": 3138.50671453,
            "unit": "iter/sec",
            "range": "stddev: 0.00000772269380338",
            "extra": "mean: 318.622864616 usec\nrounds: 2600"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[eager]",
            "value": 567.978232671,
            "unit": "iter/sec",
            "range": "stddev: 0.000131089898839",
            "extra": "mean: 1.76063085252 msec\nrounds: 556"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile]",
            "value": 1228.13127547,
            "unit": "iter/sec",
            "range": "stddev: 0.0000506950704782",
            "extra": "mean: 814.245203242 usec\nrounds: 925"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile-overhead]",
            "value": 1200.50764146,
            "unit": "iter/sec",
            "range": "stddev: 0.0000561676625147",
            "extra": "mean: 832.980953613 usec\nrounds: 927"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 159098.433684,
            "unit": "iter/sec",
            "range": "stddev: 0.00000110278631919",
            "extra": "mean: 6.28541700155 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 86272.7315708,
            "unit": "iter/sec",
            "range": "stddev: 7.10653815437e-7",
            "extra": "mean: 11.591148 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 112155.012585,
            "unit": "iter/sec",
            "range": "stddev: 8.59739057548e-7",
            "extra": "mean: 8.91623099983 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 64627.2670907,
            "unit": "iter/sec",
            "range": "stddev: 0.00000107748928968",
            "extra": "mean: 15.4733449982 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_tdmodules",
            "value": 12424.4128233,
            "unit": "iter/sec",
            "range": "stddev: 0.00000382512562439",
            "extra": "mean: 80.4867009994 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_selected_out_keys",
            "value": 16675.3389537,
            "unit": "iter/sec",
            "range": "stddev: 0.00000325170103194",
            "extra": "mean: 59.9687960033 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_probseq",
            "value": 17102.1135563,
            "unit": "iter/sec",
            "range": "stddev: 0.00000310930952258",
            "extra": "mean: 58.4723049993 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 649.289791958,
            "unit": "iter/sec",
            "range": "stddev: 0.0000286414200451",
            "extra": "mean: 1.54014434292 msec\nrounds: 452"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 9752.21603779,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031404051093",
            "extra": "mean: 102.540796484 usec\nrounds: 4039"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 12594.5936065,
            "unit": "iter/sec",
            "range": "stddev: 0.00000244246300846",
            "extra": "mean: 79.3991478603 usec\nrounds: 5796"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 6064.58613997,
            "unit": "iter/sec",
            "range": "stddev: 0.00000425366633735",
            "extra": "mean: 164.891713452 usec\nrounds: 3427"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_soft_update_lerp",
            "value": 7282.47742301,
            "unit": "iter/sec",
            "range": "stddev: 0.00000538950131492",
            "extra": "mean: 137.315908023 usec\nrounds: 3153"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1710.23578486,
            "unit": "iter/sec",
            "range": "stddev: 0.0000598248339032",
            "extra": "mean: 584.714697735 usec\nrounds: 1588"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1686.5695281,
            "unit": "iter/sec",
            "range": "stddev: 0.000214096101462",
            "extra": "mean: 592.919522936 usec\nrounds: 1308"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1995.02169722,
            "unit": "iter/sec",
            "range": "stddev: 0.0000310667547531",
            "extra": "mean: 501.247681363 usec\nrounds: 1701"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1909.45665205,
            "unit": "iter/sec",
            "range": "stddev: 0.0000572797361555",
            "extra": "mean: 523.709191789 usec\nrounds: 1778"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 623.930320464,
            "unit": "iter/sec",
            "range": "stddev: 0.0000133383889597",
            "extra": "mean: 1.60274307435 msec\nrounds: 538"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 623.775594813,
            "unit": "iter/sec",
            "range": "stddev: 0.0000126868343433",
            "extra": "mean: 1.60314062992 msec\nrounds: 508"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_from_module_eager[4-False]",
            "value": 36635.6592641,
            "unit": "iter/sec",
            "range": "stddev: 0.00000111700592715",
            "extra": "mean: 27.2958101502 usec\nrounds: 7743"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_from_module_eager[4-True]",
            "value": 5672.22660914,
            "unit": "iter/sec",
            "range": "stddev: 0.00000403011713394",
            "extra": "mean: 176.297610957 usec\nrounds: 2519"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_from_module_eager[64-False]",
            "value": 2453.10022145,
            "unit": "iter/sec",
            "range": "stddev: 0.00000451012841467",
            "extra": "mean: 407.647429671 usec\nrounds: 1038"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_from_module_eager[64-True]",
            "value": 407.908329123,
            "unit": "iter/sec",
            "range": "stddev: 0.0000299355833121",
            "extra": "mean: 2.45153121083 msec\nrounds: 351"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_from_module_eager[transformer-False]",
            "value": 9879.21443938,
            "unit": "iter/sec",
            "range": "stddev: 0.00000223750188479",
            "extra": "mean: 101.222623128 usec\nrounds: 7143"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_from_module_eager[transformer-True]",
            "value": 4172.40309472,
            "unit": "iter/sec",
            "range": "stddev: 0.00000422844315613",
            "extra": "mean: 239.67003602 usec\nrounds: 2693"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_eager[4-no_swap]",
            "value": 13394.0352145,
            "unit": "iter/sec",
            "range": "stddev: 0.00000203920743429",
            "extra": "mean: 74.6600993642 usec\nrounds: 7699"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_eager[4-swap]",
            "value": 11546.6616022,
            "unit": "iter/sec",
            "range": "stddev: 0.00000207674611344",
            "extra": "mean: 86.6051188174 usec\nrounds: 6935"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_eager[4-context]",
            "value": 5838.87442855,
            "unit": "iter/sec",
            "range": "stddev: 0.00000315342609821",
            "extra": "mean: 171.265885615 usec\nrounds: 4345"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_eager[64-no_swap]",
            "value": 855.84104712,
            "unit": "iter/sec",
            "range": "stddev: 0.0000100657743491",
            "extra": "mean: 1.16844126998 msec\nrounds: 663"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_eager[64-swap]",
            "value": 735.682355078,
            "unit": "iter/sec",
            "range": "stddev: 0.0000125823689222",
            "extra": "mean: 1.359282295 msec\nrounds: 600"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_eager[64-context]",
            "value": 375.824857355,
            "unit": "iter/sec",
            "range": "stddev: 0.0000203661062888",
            "extra": "mean: 2.66081388825 msec\nrounds: 349"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_eager[transformer-no_swap]",
            "value": 3835.92543751,
            "unit": "iter/sec",
            "range": "stddev: 0.00000383368113205",
            "extra": "mean: 260.693284135 usec\nrounds: 3076"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_eager[transformer-swap]",
            "value": 3395.07491109,
            "unit": "iter/sec",
            "range": "stddev: 0.00000451667413827",
            "extra": "mean: 294.544310859 usec\nrounds: 2551"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_eager[transformer-context]",
            "value": 1737.46385703,
            "unit": "iter/sec",
            "range": "stddev: 0.00000638367396496",
            "extra": "mean: 575.551540801 usec\nrounds: 1544"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_context_small[False-True]",
            "value": 16633.5397626,
            "unit": "iter/sec",
            "range": "stddev: 0.00000184854876801",
            "extra": "mean: 60.1194943634 usec\nrounds: 9402"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_context_small[False-False]",
            "value": 37288.0022601,
            "unit": "iter/sec",
            "range": "stddev: 0.00000113469508863",
            "extra": "mean: 26.8182777137 usec\nrounds: 18285"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_context_small[True-True]",
            "value": 15531.7332291,
            "unit": "iter/sec",
            "range": "stddev: 0.00000195096729421",
            "extra": "mean: 64.3843146963 usec\nrounds: 9676"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_context_small[True-False]",
            "value": 31720.958692,
            "unit": "iter/sec",
            "range": "stddev: 0.00000124382585821",
            "extra": "mean: 31.5248984028 usec\nrounds: 16526"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_inplace_copy[False-small]",
            "value": 9940.13499341,
            "unit": "iter/sec",
            "range": "stddev: 0.00000239419917264",
            "extra": "mean: 100.602255469 usec\nrounds: 7085"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_inplace_copy[False-many_leaves]",
            "value": 618.218625806,
            "unit": "iter/sec",
            "range": "stddev: 0.0000111023511786",
            "extra": "mean: 1.61755074703 msec\nrounds: 589"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_inplace_copy[False-16MiB]",
            "value": 1063.51591585,
            "unit": "iter/sec",
            "range": "stddev: 0.0000707743824405",
            "extra": "mean: 940.277418607 usec\nrounds: 817"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_inplace_copy[True-small]",
            "value": 6039.65423342,
            "unit": "iter/sec",
            "range": "stddev: 0.00000355035607752",
            "extra": "mean: 165.572392285 usec\nrounds: 3992"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_inplace_copy[True-many_leaves]",
            "value": 377.492237962,
            "unit": "iter/sec",
            "range": "stddev: 0.0000164443087494",
            "extra": "mean: 2.64906109169 msec\nrounds: 349"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_inplace_copy[True-16MiB]",
            "value": 511.18643667,
            "unit": "iter/sec",
            "range": "stddev: 0.000260471927823",
            "extra": "mean: 1.95623343709 msec\nrounds: 453"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_plain_tensor_speed[None]",
            "value": 626.895054296,
            "unit": "iter/sec",
            "range": "stddev: 0.0000180134599974",
            "extra": "mean: 1.59516332622 msec\nrounds: 328"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_plain_tensor_speed[False]",
            "value": 1345.87137509,
            "unit": "iter/sec",
            "range": "stddev: 0.0000196505109733",
            "extra": "mean: 743.013053484 usec\nrounds: 804"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_plain_tensor_speed[True]",
            "value": 623.853428217,
            "unit": "iter/sec",
            "range": "stddev: 0.0000139891506066",
            "extra": "mean: 1.60294061837 msec\nrounds: 283"
          },
          {
            "name": "benchmarks/storage/test_h5_benchmarks.py::test_h5_index_many_keys[slice]",
            "value": 7.59287870474,
            "unit": "iter/sec",
            "range": "stddev: 0.00030309547572",
            "extra": "mean: 131.702354125 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/storage/test_h5_benchmarks.py::test_h5_index_many_keys[sorted_fancy]",
            "value": 5.63681483435,
            "unit": "iter/sec",
            "range": "stddev: 0.000187512220974",
            "extra": "mean: 177.405153333 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/storage/test_h5_benchmarks.py::test_h5_index_many_keys[unsorted_fancy]",
            "value": 5.64996608603,
            "unit": "iter/sec",
            "range": "stddev: 0.000278219629441",
            "extra": "mean: 176.992212833 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/storage/test_h5_benchmarks.py::test_h5_index_large_arrays[slice]",
            "value": 208.542379968,
            "unit": "iter/sec",
            "range": "stddev: 0.0000303564413727",
            "extra": "mean: 4.79518839362 msec\nrounds: 188"
          },
          {
            "name": "benchmarks/storage/test_h5_benchmarks.py::test_h5_index_large_arrays[sorted_fancy]",
            "value": 36.8286565473,
            "unit": "iter/sec",
            "range": "stddev: 0.00144025849154",
            "extra": "mean: 27.1527688966 msec\nrounds: 29"
          },
          {
            "name": "benchmarks/storage/test_h5_benchmarks.py::test_h5_index_large_arrays[dense_fancy]",
            "value": 9.620805746,
            "unit": "iter/sec",
            "range": "stddev: 0.00614350986133",
            "extra": "mean: 103.941398091 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/storage/test_h5_benchmarks.py::test_h5_index_large_arrays[dense_mask]",
            "value": 9.13644643111,
            "unit": "iter/sec",
            "range": "stddev: 0.00675604567689",
            "extra": "mean: 109.451744455 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/storage/test_h5_benchmarks.py::test_h5_to_tensordict_many_keys",
            "value": 11.9405343954,
            "unit": "iter/sec",
            "range": "stddev: 0.000177542917393",
            "extra": "mean: 83.7483455 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/storage/test_h5_benchmarks.py::test_h5_index_string_column[int]",
            "value": 44.8957702996,
            "unit": "iter/sec",
            "range": "stddev: 0.000268766431985",
            "extra": "mean: 22.273813175 msec\nrounds: 40"
          },
          {
            "name": "benchmarks/storage/test_h5_benchmarks.py::test_h5_index_string_column[slice]",
            "value": 45.046989596,
            "unit": "iter/sec",
            "range": "stddev: 0.0000551676575194",
            "extra": "mean: 22.1990416889 msec\nrounds: 45"
          },
          {
            "name": "benchmarks/storage/test_h5_benchmarks.py::test_h5_index_string_column[sorted_fancy]",
            "value": 44.7625921922,
            "unit": "iter/sec",
            "range": "stddev: 0.000166420022837",
            "extra": "mean: 22.3400824444 msec\nrounds: 45"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init",
            "value": 67553.8810251,
            "unit": "iter/sec",
            "range": "stddev: 9.16725219035e-7",
            "extra": "mean: 14.8029985077 usec\nrounds: 6701"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_only",
            "value": 156051.747859,
            "unit": "iter/sec",
            "range": "stddev: 5.76189160867e-7",
            "extra": "mean: 6.40813072408 usec\nrounds: 23737"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_ttd_init",
            "value": 283404.814889,
            "unit": "iter/sec",
            "range": "stddev: 4.87230413472e-7",
            "extra": "mean: 3.52852156161 usec\nrounds: 30610"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_td_init_reference",
            "value": 532483.847132,
            "unit": "iter/sec",
            "range": "stddev: 2.68203536602e-7",
            "extra": "mean: 1.87799123933 usec\nrounds: 81500"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_nested",
            "value": 32870.7850892,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011668589833",
            "extra": "mean: 30.4221513812 usec\nrounds: 5397"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size0-False-4]",
            "value": 197019.548411,
            "unit": "iter/sec",
            "range": "stddev: 4.71675612028e-7",
            "extra": "mean: 5.07563847377 usec\nrounds: 44207"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size0-False-16]",
            "value": 121198.645998,
            "unit": "iter/sec",
            "range": "stddev: 5.42196835501e-7",
            "extra": "mean: 8.25091725876 usec\nrounds: 42772"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size0-False-64]",
            "value": 49140.2386468,
            "unit": "iter/sec",
            "range": "stddev: 9.36882341386e-7",
            "extra": "mean: 20.3499215213 usec\nrounds: 25026"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size0-False-256]",
            "value": 14224.0453898,
            "unit": "iter/sec",
            "range": "stddev: 0.00000196254741869",
            "extra": "mean: 70.3034876923 usec\nrounds: 10603"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size0-True-4]",
            "value": 195768.724698,
            "unit": "iter/sec",
            "range": "stddev: 9.42588003146e-7",
            "extra": "mean: 5.10806821437 usec\nrounds: 52247"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size0-True-16]",
            "value": 119957.610784,
            "unit": "iter/sec",
            "range": "stddev: 0.00000154887968721",
            "extra": "mean: 8.33627806912 usec\nrounds: 44248"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size0-True-64]",
            "value": 48815.5522977,
            "unit": "iter/sec",
            "range": "stddev: 9.56759529581e-7",
            "extra": "mean: 20.4852747317 usec\nrounds: 25221"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size0-True-256]",
            "value": 14082.1346964,
            "unit": "iter/sec",
            "range": "stddev: 0.00000208167977015",
            "extra": "mean: 71.0119610099 usec\nrounds: 10464"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size1-False-4]",
            "value": 59518.2067258,
            "unit": "iter/sec",
            "range": "stddev: 8.1413283135e-7",
            "extra": "mean: 16.8015814826 usec\nrounds: 21839"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size1-False-16]",
            "value": 18913.3664665,
            "unit": "iter/sec",
            "range": "stddev: 0.00000170141281979",
            "extra": "mean: 52.8726602835 usec\nrounds: 11489"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size1-False-64]",
            "value": 5112.30836603,
            "unit": "iter/sec",
            "range": "stddev: 0.00000332437315638",
            "extra": "mean: 195.606353999 usec\nrounds: 4500"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size1-False-256]",
            "value": 1312.68795024,
            "unit": "iter/sec",
            "range": "stddev: 0.00000742124173483",
            "extra": "mean: 761.795672625 usec\nrounds: 1231"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size1-True-4]",
            "value": 59864.8497178,
            "unit": "iter/sec",
            "range": "stddev: 9.33379512332e-7",
            "extra": "mean: 16.7042931656 usec\nrounds: 24771"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size1-True-16]",
            "value": 18684.4979943,
            "unit": "iter/sec",
            "range": "stddev: 0.00000174657141205",
            "extra": "mean: 53.5203033181 usec\nrounds: 12116"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size1-True-64]",
            "value": 5098.39132472,
            "unit": "iter/sec",
            "range": "stddev: 0.00000595658173067",
            "extra": "mean: 196.140299226 usec\nrounds: 3870"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[None-batch_size1-True-256]",
            "value": 1297.67840596,
            "unit": "iter/sec",
            "range": "stddev: 0.00000737822423723",
            "extra": "mean: 770.606951158 usec\nrounds: 1208"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size0-False-4]",
            "value": 76520.7101276,
            "unit": "iter/sec",
            "range": "stddev: 8.59627711591e-7",
            "extra": "mean: 13.068357551 usec\nrounds: 17687"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size0-False-16]",
            "value": 31397.853504,
            "unit": "iter/sec",
            "range": "stddev: 0.00000132979326569",
            "extra": "mean: 31.8493109685 usec\nrounds: 16127"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size0-False-64]",
            "value": 9350.53607501,
            "unit": "iter/sec",
            "range": "stddev: 0.00000275274720394",
            "extra": "mean: 106.945740007 usec\nrounds: 6804"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size0-False-256]",
            "value": 2517.37858947,
            "unit": "iter/sec",
            "range": "stddev: 0.00000842511668504",
            "extra": "mean: 397.238621232 usec\nrounds: 2223"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size0-True-4]",
            "value": 77990.3910277,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010829085508",
            "extra": "mean: 12.8220923991 usec\nrounds: 24589"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size0-True-16]",
            "value": 31355.1554997,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014147532672",
            "extra": "mean: 31.8926818912 usec\nrounds: 16290"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size0-True-64]",
            "value": 9371.73071211,
            "unit": "iter/sec",
            "range": "stddev: 0.00000828287816611",
            "extra": "mean: 106.703876874 usec\nrounds: 6936"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size0-True-256]",
            "value": 2490.12893604,
            "unit": "iter/sec",
            "range": "stddev: 0.0000055906537095",
            "extra": "mean: 401.585630979 usec\nrounds: 2195"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size1-False-4]",
            "value": 41011.4472739,
            "unit": "iter/sec",
            "range": "stddev: 0.00000124622440897",
            "extra": "mean: 24.3834360032 usec\nrounds: 16743"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size1-False-16]",
            "value": 13219.1763716,
            "unit": "iter/sec",
            "range": "stddev: 0.00000233011049869",
            "extra": "mean: 75.647678183 usec\nrounds: 8452"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size1-False-64]",
            "value": 3520.34754471,
            "unit": "iter/sec",
            "range": "stddev: 0.00000458895041969",
            "extra": "mean: 284.062862345 usec\nrounds: 3131"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size1-False-256]",
            "value": 894.935079899,
            "unit": "iter/sec",
            "range": "stddev: 0.0000114595888513",
            "extra": "mean: 1.11739948792 msec\nrounds: 869"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size1-True-4]",
            "value": 41051.8070129,
            "unit": "iter/sec",
            "range": "stddev: 0.00000177229465653",
            "extra": "mean: 24.359463633 usec\nrounds: 17337"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size1-True-16]",
            "value": 13038.1589907,
            "unit": "iter/sec",
            "range": "stddev: 0.00000234371171201",
            "extra": "mean: 76.6979449103 usec\nrounds: 8804"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size1-True-64]",
            "value": 3486.81510001,
            "unit": "iter/sec",
            "range": "stddev: 0.00000479935413006",
            "extra": "mean: 286.794674027 usec\nrounds: 3034"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_tensor_schema[cpu-batch_size1-True-256]",
            "value": 903.422224388,
            "unit": "iter/sec",
            "range": "stddev: 0.0000117621545011",
            "extra": "mean: 1.1069021472 msec\nrounds: 856"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_nested_tensor_schema[False-False-4]",
            "value": 8102.02622848,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032540895874",
            "extra": "mean: 123.425914925 usec\nrounds: 4020"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_nested_tensor_schema[False-False-64]",
            "value": 534.474315087,
            "unit": "iter/sec",
            "range": "stddev: 0.0000133265155355",
            "extra": "mean: 1.87099729916 msec\nrounds: 478"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_nested_tensor_schema[False-True-4]",
            "value": 8675.63525279,
            "unit": "iter/sec",
            "range": "stddev: 0.00000330772065689",
            "extra": "mean: 115.265334568 usec\nrounds: 5401"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_nested_tensor_schema[False-True-64]",
            "value": 567.801205114,
            "unit": "iter/sec",
            "range": "stddev: 0.0000126737179668",
            "extra": "mean: 1.76117977735 msec\nrounds: 512"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_nested_tensor_schema[True-False-4]",
            "value": 44240.2084594,
            "unit": "iter/sec",
            "range": "stddev: 0.00000106582843686",
            "extra": "mean: 22.6038717904 usec\nrounds: 17643"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_nested_tensor_schema[True-False-64]",
            "value": 3935.00196337,
            "unit": "iter/sec",
            "range": "stddev: 0.00000414879040563",
            "extra": "mean: 254.129479301 usec\nrounds: 3261"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_nested_tensor_schema[True-True-4]",
            "value": 64794.064796,
            "unit": "iter/sec",
            "range": "stddev: 8.71470084602e-7",
            "extra": "mean: 15.4335123618 usec\nrounds: 26897"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_nested_tensor_schema[True-True-64]",
            "value": 6899.15901023,
            "unit": "iter/sec",
            "range": "stddev: 0.00000277203944446",
            "extra": "mean: 144.945202527 usec\nrounds: 5303"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_init_many_fields",
            "value": 152042.481116,
            "unit": "iter/sec",
            "range": "stddev: 6.16495671695e-7",
            "extra": "mean: 6.57710919121 usec\nrounds: 26275"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_first_layer_tensor",
            "value": 2467021.2215,
            "unit": "iter/sec",
            "range": "stddev: 2.34615240283e-7",
            "extra": "mean: 405.347141437 nsec\nrounds: 191939"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_first_layer_tensor_only",
            "value": 4981953.65867,
            "unit": "iter/sec",
            "range": "stddev: 2.01342469948e-8",
            "extra": "mean: 200.724468454 nsec\nrounds: 191939"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_ttd_first_layer_tensor",
            "value": 2158030.40138,
            "unit": "iter/sec",
            "range": "stddev: 1.48262706399e-7",
            "extra": "mean: 463.385501595 nsec\nrounds: 110853"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_td_first_layer_tensor_reference",
            "value": 5016757.46768,
            "unit": "iter/sec",
            "range": "stddev: 1.98479523781e-8",
            "extra": "mean: 199.331940291 nsec\nrounds: 199204"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_first_layer_tensor_set",
            "value": 510519.65742,
            "unit": "iter/sec",
            "range": "stddev: 2.90525696896e-7",
            "extra": "mean: 1.95878843343 usec\nrounds: 118204"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_first_layer_tensor_only_set",
            "value": 897945.389745,
            "unit": "iter/sec",
            "range": "stddev: 2.65092467532e-7",
            "extra": "mean: 1.11365347093 usec\nrounds: 99602"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_first_layer_nontensor",
            "value": 729997.136938,
            "unit": "iter/sec",
            "range": "stddev: 2.31649434777e-7",
            "extra": "mean: 1.36986838633 usec\nrounds: 94162"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_second_layer_tensor",
            "value": 1779357.82036,
            "unit": "iter/sec",
            "range": "stddev: 4.12678819835e-8",
            "extra": "mean: 562.000508588 nsec\nrounds: 80322"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_second_layer_nontensor",
            "value": 618385.21715,
            "unit": "iter/sec",
            "range": "stddev: 2.65786230883e-7",
            "extra": "mean: 1.61711498313 usec\nrounds: 137742"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_tc_clone",
            "value": 56786.5941844,
            "unit": "iter/sec",
            "range": "stddev: 0.00000142181099145",
            "extra": "mean: 17.6097900281 usec\nrounds: 11573"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_nontensordata_init",
            "value": 258471.542106,
            "unit": "iter/sec",
            "range": "stddev: 4.5110113024e-7",
            "extra": "mean: 3.86889787498 usec\nrounds: 42536"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_nontensordata_data",
            "value": 1197936.34182,
            "unit": "iter/sec",
            "range": "stddev: 1.8313757337e-7",
            "extra": "mean: 834.768898056 nsec\nrounds: 105486"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_nontensordata_copy",
            "value": 356668.057495,
            "unit": "iter/sec",
            "range": "stddev: 4.25371388847e-7",
            "extra": "mean: 2.80372738457 usec\nrounds: 57704"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_nontensor_lazy_stack_copy",
            "value": 88.1571341473,
            "unit": "iter/sec",
            "range": "stddev: 0.0513497963701",
            "extra": "mean: 11.343381448 msec\nrounds: 154"
          },
          {
            "name": "benchmarks/tensorclass/test_tensorclass_speed.py::test_nontensor_stack_of_dicts_copy",
            "value": 616.487672667,
            "unit": "iter/sec",
            "range": "stddev: 0.0153926042737",
            "extra": "mean: 1.62209245105 msec\nrounds: 858"
          },
          {
            "name": "benchmarks/tensorclass/test_torch_functions.py::test_unbind",
            "value": 90.0166600486,
            "unit": "iter/sec",
            "range": "stddev: 0.0494019010775",
            "extra": "mean: 11.1090546956 msec\nrounds: 161"
          },
          {
            "name": "benchmarks/tensorclass/test_torch_functions.py::test_full_like",
            "value": 3148.01269531,
            "unit": "iter/sec",
            "range": "stddev: 0.0000529936713472",
            "extra": "mean: 317.660726556 usec\nrounds: 1141"
          },
          {
            "name": "benchmarks/tensorclass/test_torch_functions.py::test_zeros_like",
            "value": 2936.79822959,
            "unit": "iter/sec",
            "range": "stddev: 0.0000895295722132",
            "extra": "mean: 340.5068792 usec\nrounds: 1399"
          },
          {
            "name": "benchmarks/tensorclass/test_torch_functions.py::test_ones_like",
            "value": 3196.35950865,
            "unit": "iter/sec",
            "range": "stddev: 0.0000714186795919",
            "extra": "mean: 312.855921649 usec\nrounds: 1940"
          },
          {
            "name": "benchmarks/tensorclass/test_torch_functions.py::test_clone",
            "value": 1601.64332901,
            "unit": "iter/sec",
            "range": "stddev: 0.000103244584421",
            "extra": "mean: 624.358733238 usec\nrounds: 2088"
          },
          {
            "name": "benchmarks/tensorclass/test_torch_functions.py::test_squeeze",
            "value": 117176.712849,
            "unit": "iter/sec",
            "range": "stddev: 6.20404870352e-7",
            "extra": "mean: 8.534118902 usec\nrounds: 18595"
          },
          {
            "name": "benchmarks/tensorclass/test_torch_functions.py::test_unsqueeze",
            "value": 19429.4328098,
            "unit": "iter/sec",
            "range": "stddev: 0.00000310749363631",
            "extra": "mean: 51.4683063468 usec\nrounds: 4743"
          },
          {
            "name": "benchmarks/tensorclass/test_torch_functions.py::test_split",
            "value": 10840.3849205,
            "unit": "iter/sec",
            "range": "stddev: 0.00000317439288106",
            "extra": "mean: 92.2476468623 usec\nrounds: 3888"
          },
          {
            "name": "benchmarks/tensorclass/test_torch_functions.py::test_permute",
            "value": 15165.5907655,
            "unit": "iter/sec",
            "range": "stddev: 0.00000247623519868",
            "extra": "mean: 65.9387435323 usec\nrounds: 4445"
          },
          {
            "name": "benchmarks/tensorclass/test_torch_functions.py::test_stack",
            "value": 319.919257828,
            "unit": "iter/sec",
            "range": "stddev: 0.000389719734038",
            "extra": "mean: 3.12578869677 msec\nrounds: 310"
          },
          {
            "name": "benchmarks/tensorclass/test_torch_functions.py::test_cat",
            "value": 321.417655826,
            "unit": "iter/sec",
            "range": "stddev: 0.000327755003817",
            "extra": "mean: 3.11121676695 msec\nrounds: 236"
          },
          {
            "name": "benchmarks/test_fx_benchmarks.py::TestFxSequential::test_sequential_tensordict",
            "value": 4305.92446605,
            "unit": "iter/sec",
            "range": "stddev: 0.0000390537982307",
            "extra": "mean: 232.238165784 usec\nrounds: 1701"
          },
          {
            "name": "benchmarks/test_fx_benchmarks.py::TestFxSequential::test_sequential_graph_module",
            "value": 5226.49570789,
            "unit": "iter/sec",
            "range": "stddev: 0.0000292378797941",
            "extra": "mean: 191.332788907 usec\nrounds: 3245"
          },
          {
            "name": "benchmarks/test_fx_benchmarks.py::TestFxNested::test_nested_tensordict",
            "value": 3196.64767444,
            "unit": "iter/sec",
            "range": "stddev: 0.0000419238236771",
            "extra": "mean: 312.827718862 usec\nrounds: 2216"
          },
          {
            "name": "benchmarks/test_fx_benchmarks.py::TestFxNested::test_nested_graph_module",
            "value": 3861.40827082,
            "unit": "iter/sec",
            "range": "stddev: 0.0000437070778871",
            "extra": "mean: 258.972874626 usec\nrounds: 3015"
          }
        ]
      }
    ]
  }
}