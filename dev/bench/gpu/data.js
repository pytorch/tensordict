window.BENCHMARK_DATA = {
  "lastUpdate": 1792423883188,
  "repoUrl": "https://github.com/pytorch/tensordict",
  "entries": {
    "GPU Benchmark Results": [
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
        "date": 1792423588368,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-2]",
            "value": 4470.43228366,
            "unit": "iter/sec",
            "range": "stddev: 0.00000630188884159",
            "extra": "mean: 223.69201378 usec\nrounds: 146"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-8]",
            "value": 3564.18282094,
            "unit": "iter/sec",
            "range": "stddev: 0.00000916992105703",
            "extra": "mean: 280.569221681 usec\nrounds: 3142"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-32]",
            "value": 2020.24037155,
            "unit": "iter/sec",
            "range": "stddev: 0.00000873192192768",
            "extra": "mean: 494.990603139 usec\nrounds: 1823"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-2]",
            "value": 4465.83799519,
            "unit": "iter/sec",
            "range": "stddev: 0.00000542168939997",
            "extra": "mean: 223.922139826 usec\nrounds: 3943"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-8]",
            "value": 3568.3023867,
            "unit": "iter/sec",
            "range": "stddev: 0.00000627384804462",
            "extra": "mean: 280.245307608 usec\nrounds: 3200"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-32]",
            "value": 1797.77620155,
            "unit": "iter/sec",
            "range": "stddev: 0.00000780672422989",
            "extra": "mean: 556.242762106 usec\nrounds: 1647"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-2]",
            "value": 2253.21371023,
            "unit": "iter/sec",
            "range": "stddev: 0.000011085359006",
            "extra": "mean: 443.810542897 usec\nrounds: 72"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-8]",
            "value": 601.55145162,
            "unit": "iter/sec",
            "range": "stddev: 0.0000259461187788",
            "extra": "mean: 1.66236819362 msec\nrounds: 588"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-32]",
            "value": 152.669904877,
            "unit": "iter/sec",
            "range": "stddev: 0.000055923411715",
            "extra": "mean: 6.55007940698 msec\nrounds: 150"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-2]",
            "value": 2276.84699854,
            "unit": "iter/sec",
            "range": "stddev: 0.00000870178047869",
            "extra": "mean: 439.203864222 usec\nrounds: 2113"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-8]",
            "value": 601.288929795,
            "unit": "iter/sec",
            "range": "stddev: 0.0000170665221818",
            "extra": "mean: 1.66309398103 msec\nrounds: 590"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-32]",
            "value": 151.73737279,
            "unit": "iter/sec",
            "range": "stddev: 0.0000444975479048",
            "extra": "mean: 6.59033421767 msec\nrounds: 152"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-2]",
            "value": 3078.62154481,
            "unit": "iter/sec",
            "range": "stddev: 0.00000978597064302",
            "extra": "mean: 324.820698304 usec\nrounds: 2352"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-8]",
            "value": 831.190503008,
            "unit": "iter/sec",
            "range": "stddev: 0.0000143547710107",
            "extra": "mean: 1.20309363062 msec\nrounds: 806"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-32]",
            "value": 212.963118417,
            "unit": "iter/sec",
            "range": "stddev: 0.0000317829521213",
            "extra": "mean: 4.69564874628 msec\nrounds: 205"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-2]",
            "value": 3063.05729635,
            "unit": "iter/sec",
            "range": "stddev: 0.00000721592268756",
            "extra": "mean: 326.47120287 usec\nrounds: 2888"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-8]",
            "value": 823.992310233,
            "unit": "iter/sec",
            "range": "stddev: 0.0000195231765378",
            "extra": "mean: 1.21360355865 msec\nrounds: 809"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-32]",
            "value": 211.311961024,
            "unit": "iter/sec",
            "range": "stddev: 0.0000265628839384",
            "extra": "mean: 4.73233978405 msec\nrounds: 208"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-2]",
            "value": 3557.89955232,
            "unit": "iter/sec",
            "range": "stddev: 0.0000116431332024",
            "extra": "mean: 281.064708346 usec\nrounds: 3021"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-8]",
            "value": 955.01510043,
            "unit": "iter/sec",
            "range": "stddev: 0.0000131434693438",
            "extra": "mean: 1.04710386207 msec\nrounds: 900"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-32]",
            "value": 243.714303207,
            "unit": "iter/sec",
            "range": "stddev: 0.0000417080095182",
            "extra": "mean: 4.10316500443 msec\nrounds: 242"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-2]",
            "value": 3575.62543008,
            "unit": "iter/sec",
            "range": "stddev: 0.00000741497271142",
            "extra": "mean: 279.671352482 usec\nrounds: 3277"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-8]",
            "value": 954.938665037,
            "unit": "iter/sec",
            "range": "stddev: 0.0000155372265955",
            "extra": "mean: 1.04718767457 msec\nrounds: 925"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-32]",
            "value": 243.291656544,
            "unit": "iter/sec",
            "range": "stddev: 0.0000444751796431",
            "extra": "mean: 4.11029302938 msec\nrounds: 242"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-128]",
            "value": 189.365375095,
            "unit": "iter/sec",
            "range": "stddev: 0.0000511081212292",
            "extra": "mean: 5.28079644706 msec\nrounds: 186"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-256]",
            "value": 100.135330412,
            "unit": "iter/sec",
            "range": "stddev: 0.0000242495706722",
            "extra": "mean: 9.98648524839 msec\nrounds: 97"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-128]",
            "value": 182.363221615,
            "unit": "iter/sec",
            "range": "stddev: 0.0000432894394538",
            "extra": "mean: 5.48356182318 msec\nrounds: 181"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-256]",
            "value": 95.5113041644,
            "unit": "iter/sec",
            "range": "stddev: 0.0000373063575336",
            "extra": "mean: 10.4699648774 msec\nrounds: 90"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-128]",
            "value": 13.0583492752,
            "unit": "iter/sec",
            "range": "stddev: 0.000423173305916",
            "extra": "mean: 76.579357691 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-256]",
            "value": 6.49073973449,
            "unit": "iter/sec",
            "range": "stddev: 0.00296825403276",
            "extra": "mean: 154.065644427 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-128]",
            "value": 12.5581526635,
            "unit": "iter/sec",
            "range": "stddev: 0.00187699411351",
            "extra": "mean: 79.6295463828 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-256]",
            "value": 6.23863344932,
            "unit": "iter/sec",
            "range": "stddev: 0.00220676137504",
            "extra": "mean: 160.291513858 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-128]",
            "value": 23.3108500131,
            "unit": "iter/sec",
            "range": "stddev: 0.000085361257135",
            "extra": "mean: 42.8984785813 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-256]",
            "value": 11.7012340526,
            "unit": "iter/sec",
            "range": "stddev: 0.000270788660455",
            "extra": "mean: 85.4610714996 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-128]",
            "value": 23.1954191722,
            "unit": "iter/sec",
            "range": "stddev: 0.000155323757307",
            "extra": "mean: 43.1119607098 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-256]",
            "value": 11.6535498791,
            "unit": "iter/sec",
            "range": "stddev: 0.000104012071662",
            "extra": "mean: 85.8107624178 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-128]",
            "value": 27.0494872259,
            "unit": "iter/sec",
            "range": "stddev: 0.000151162190909",
            "extra": "mean: 36.9692775189 msec\nrounds: 27"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-256]",
            "value": 13.4929228703,
            "unit": "iter/sec",
            "range": "stddev: 0.00036377007836",
            "extra": "mean: 74.112926429 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-128]",
            "value": 27.1745480183,
            "unit": "iter/sec",
            "range": "stddev: 0.000192820047311",
            "extra": "mean: 36.7991401118 msec\nrounds: 27"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-256]",
            "value": 13.4993729467,
            "unit": "iter/sec",
            "range": "stddev: 0.000120407039313",
            "extra": "mean: 74.0775148557 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_mask_many_leaves[cuda]",
            "value": 6511.45237473,
            "unit": "iter/sec",
            "range": "stddev: 0.0000175845493006",
            "extra": "mean: 153.575568468 usec\nrounds: 44"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-None]",
            "value": 6811.45349784,
            "unit": "iter/sec",
            "range": "stddev: 0.00000605573166107",
            "extra": "mean: 146.81154328 usec\nrounds: 1441"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-cpu]",
            "value": 6925.33608021,
            "unit": "iter/sec",
            "range": "stddev: 0.00000540948948428",
            "extra": "mean: 144.39732432 usec\nrounds: 5111"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to",
            "value": 23147.3163773,
            "unit": "iter/sec",
            "range": "stddev: 0.00000259183241634",
            "extra": "mean: 43.2015523397 usec\nrounds: 11836"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_nonblocking",
            "value": 24391.1977355,
            "unit": "iter/sec",
            "range": "stddev: 0.00000248625651676",
            "extra": "mean: 40.9983966694 usec\nrounds: 12763"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[add]",
            "value": 10952.8459605,
            "unit": "iter/sec",
            "range": "stddev: 0.00000768147195917",
            "extra": "mean: 91.3004714579 usec\nrounds: 189"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[sub]",
            "value": 11245.1681921,
            "unit": "iter/sec",
            "range": "stddev: 0.00000383804475682",
            "extra": "mean: 88.9270825405 usec\nrounds: 8216"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_blocking",
            "value": 55.2853086261,
            "unit": "iter/sec",
            "range": "stddev: 0.0000682883505563",
            "extra": "mean: 18.087988018 msec\nrounds: 56"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_global_sync",
            "value": 92.4955451975,
            "unit": "iter/sec",
            "range": "stddev: 0.0122296192668",
            "extra": "mean: 10.8113314848 msec\nrounds: 105"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_event_sync",
            "value": 94.4640935876,
            "unit": "iter/sec",
            "range": "stddev: 0.000104071564002",
            "extra": "mean: 10.5860328726 msec\nrounds: 95"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_default",
            "value": 92.4710143816,
            "unit": "iter/sec",
            "range": "stddev: 0.000100165676463",
            "extra": "mean: 10.8141995271 msec\nrounds: 91"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_cpu_to_cpu_default",
            "value": 3861.7842919,
            "unit": "iter/sec",
            "range": "stddev: 0.00000553180534222",
            "extra": "mean: 258.947658495 usec\nrounds: 3956"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[False-None]",
            "value": 293.819959746,
            "unit": "iter/sec",
            "range": "stddev: 0.00003005888441",
            "extra": "mean: 3.4034447519 msec\nrounds: 291"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[default-None]",
            "value": 445.97863484,
            "unit": "iter/sec",
            "range": "stddev: 0.000204067076911",
            "extra": "mean: 2.24225987946 msec\nrounds: 448"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[reduce-overhead-None]",
            "value": 443.780322456,
            "unit": "iter/sec",
            "range": "stddev: 0.000263395222897",
            "extra": "mean: 2.25336714901 msec\nrounds: 409"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[False-False-None]",
            "value": 474.65927547,
            "unit": "iter/sec",
            "range": "stddev: 0.0000554499816264",
            "extra": "mean: 2.10677437834 msec\nrounds: 442"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-False-None]",
            "value": 636.566342277,
            "unit": "iter/sec",
            "range": "stddev: 0.000043705584611",
            "extra": "mean: 1.57092817133 msec\nrounds: 626"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[within-False-None]",
            "value": 192.780085311,
            "unit": "iter/sec",
            "range": "stddev: 0.000047461631285",
            "extra": "mean: 5.18725779371 msec\nrounds: 194"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-default-None]",
            "value": 117.435303328,
            "unit": "iter/sec",
            "range": "stddev: 0.000125824351352",
            "extra": "mean: 8.51532692183 msec\nrounds: 115"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[False-False-None]",
            "value": 141.026364109,
            "unit": "iter/sec",
            "range": "stddev: 0.0000684843046503",
            "extra": "mean: 7.09087273373 msec\nrounds: 139"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[True-False-None]",
            "value": 266.043261649,
            "unit": "iter/sec",
            "range": "stddev: 0.0000590036602198",
            "extra": "mean: 3.75878717545 msec\nrounds: 262"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[within-False-None]",
            "value": 112.132456511,
            "unit": "iter/sec",
            "range": "stddev: 0.0000981503337973",
            "extra": "mean: 8.91802454987 msec\nrounds: 111"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 10.7739374808,
            "unit": "iter/sec",
            "range": "stddev: 0.00229355053232",
            "extra": "mean: 92.8165772057 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_archive",
            "value": 4.54411710021,
            "unit": "iter/sec",
            "range": "stddev: 0.00219098965967",
            "extra": "mean: 220.064751402 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_zarr",
            "value": 0.495996592569,
            "unit": "iter/sec",
            "range": "stddev: 0.00842578930156",
            "extra": "mean: 2.0161428828 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 40.995967508,
            "unit": "iter/sec",
            "range": "stddev: 0.00106488683612",
            "extra": "mean: 24.3926430034 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.6711591012,
            "unit": "iter/sec",
            "range": "stddev: 0.0202376353984",
            "extra": "mean: 103.4002222 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 10.0856292719,
            "unit": "iter/sec",
            "range": "stddev: 0.00129728199797",
            "extra": "mean: 99.1509774001 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 43.6910690482,
            "unit": "iter/sec",
            "range": "stddev: 0.000223067724545",
            "extra": "mean: 22.8879728005 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-compile]",
            "value": 4769.71584126,
            "unit": "iter/sec",
            "range": "stddev: 0.00000710306391662",
            "extra": "mean: 209.656095516 usec\nrounds: 1693"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-eager]",
            "value": 8213.56686874,
            "unit": "iter/sec",
            "range": "stddev: 0.00000466210686092",
            "extra": "mean: 121.749784957 usec\nrounds: 6977"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-compile]",
            "value": 6492.61937735,
            "unit": "iter/sec",
            "range": "stddev: 0.00000652699388096",
            "extra": "mean: 154.021041721 usec\nrounds: 2918"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-eager]",
            "value": 5756.11411937,
            "unit": "iter/sec",
            "range": "stddev: 0.0000157165375306",
            "extra": "mean: 173.7283138 usec\nrounds: 5374"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-compile]",
            "value": 21171.3685299,
            "unit": "iter/sec",
            "range": "stddev: 0.00000242042695447",
            "extra": "mean: 47.2336022392 usec\nrounds: 16161"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-eager]",
            "value": 30337.4353344,
            "unit": "iter/sec",
            "range": "stddev: 0.00000130053754391",
            "extra": "mean: 32.9625754115 usec\nrounds: 26164"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-compile]",
            "value": 47876.8264617,
            "unit": "iter/sec",
            "range": "stddev: 0.00000433527273015",
            "extra": "mean: 20.8869316098 usec\nrounds: 38080"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-eager]",
            "value": 15724.7236807,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113468963449",
            "extra": "mean: 63.5941222438 usec\nrounds: 14925"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-compile]",
            "value": 3739.77836563,
            "unit": "iter/sec",
            "range": "stddev: 0.00000643521474941",
            "extra": "mean: 267.395525144 usec\nrounds: 1310"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-eager]",
            "value": 5338.48686307,
            "unit": "iter/sec",
            "range": "stddev: 0.0000048406216839",
            "extra": "mean: 187.318996122 usec\nrounds: 4601"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-compile]",
            "value": 5870.9212691,
            "unit": "iter/sec",
            "range": "stddev: 0.0000102574957334",
            "extra": "mean: 170.331018619 usec\nrounds: 3309"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-eager]",
            "value": 16167.4165127,
            "unit": "iter/sec",
            "range": "stddev: 0.00000599189482388",
            "extra": "mean: 61.8528012322 usec\nrounds: 13300"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-compile]",
            "value": 4063.89075858,
            "unit": "iter/sec",
            "range": "stddev: 0.0000102128847526",
            "extra": "mean: 246.069606543 usec\nrounds: 1378"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-eager]",
            "value": 1904.96465346,
            "unit": "iter/sec",
            "range": "stddev: 0.0000188304115192",
            "extra": "mean: 524.944123339 usec\nrounds: 1807"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-eager]",
            "value": 4338.49208614,
            "unit": "iter/sec",
            "range": "stddev: 0.00000841396568961",
            "extra": "mean: 230.494830956 usec\nrounds: 4031"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-compile]",
            "value": 3701.42352018,
            "unit": "iter/sec",
            "range": "stddev: 0.00000824520276944",
            "extra": "mean: 270.166327778 usec\nrounds: 1342"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-eager]",
            "value": 13418.4312103,
            "unit": "iter/sec",
            "range": "stddev: 0.00000342046125852",
            "extra": "mean: 74.5243601378 usec\nrounds: 11968"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-compile]",
            "value": 5910.16711937,
            "unit": "iter/sec",
            "range": "stddev: 0.00000769071657803",
            "extra": "mean: 169.1999532 usec\nrounds: 3287"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-eager]",
            "value": 2199.798846,
            "unit": "iter/sec",
            "range": "stddev: 0.0000223950808068",
            "extra": "mean: 454.58701909 usec\nrounds: 2160"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-compile]",
            "value": 4106.43650907,
            "unit": "iter/sec",
            "range": "stddev: 0.0000128390658691",
            "extra": "mean: 243.520141561 usec\nrounds: 1386"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-compile]",
            "value": 31417.1129004,
            "unit": "iter/sec",
            "range": "stddev: 0.00000392311365504",
            "extra": "mean: 31.8297866254 usec\nrounds: 22163"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-eager]",
            "value": 53813.5548878,
            "unit": "iter/sec",
            "range": "stddev: 0.00000194362797797",
            "extra": "mean: 18.5826786966 usec\nrounds: 48031"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-compile]",
            "value": 43005.2670896,
            "unit": "iter/sec",
            "range": "stddev: 0.00000523883640371",
            "extra": "mean: 23.2529656871 usec\nrounds: 28556"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-eager]",
            "value": 19637.2226623,
            "unit": "iter/sec",
            "range": "stddev: 0.0000117115384006",
            "extra": "mean: 50.9236981825 usec\nrounds: 15870"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-compile]",
            "value": 4132.45481901,
            "unit": "iter/sec",
            "range": "stddev: 0.000210615726165",
            "extra": "mean: 241.986916687 usec\nrounds: 96"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-eager]",
            "value": 307.01919958,
            "unit": "iter/sec",
            "range": "stddev: 0.0000632937683131",
            "extra": "mean: 3.25712529174 msec\nrounds: 288"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-compile]",
            "value": 4969.7082293,
            "unit": "iter/sec",
            "range": "stddev: 0.000228334035072",
            "extra": "mean: 201.219056303 usec\nrounds: 89"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-eager]",
            "value": 356.484471609,
            "unit": "iter/sec",
            "range": "stddev: 0.0000515317822846",
            "extra": "mean: 2.80517127572 msec\nrounds: 355"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-compile]",
            "value": 6277.56698178,
            "unit": "iter/sec",
            "range": "stddev: 0.00000729176349321",
            "extra": "mean: 159.297384305 usec\nrounds: 3872"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-eager]",
            "value": 19053.5144972,
            "unit": "iter/sec",
            "range": "stddev: 0.0000120385626311",
            "extra": "mean: 52.4837556948 usec\nrounds: 14771"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-compile]",
            "value": 7463.78255527,
            "unit": "iter/sec",
            "range": "stddev: 0.00000854293331984",
            "extra": "mean: 133.980323327 usec\nrounds: 4503"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-eager]",
            "value": 20955.7840576,
            "unit": "iter/sec",
            "range": "stddev: 0.0000122638737662",
            "extra": "mean: 47.7195220782 usec\nrounds: 17376"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-compile]",
            "value": 7378.31120173,
            "unit": "iter/sec",
            "range": "stddev: 0.00000518607764941",
            "extra": "mean: 135.532369489 usec\nrounds: 6655"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-eager]",
            "value": 21342.7765963,
            "unit": "iter/sec",
            "range": "stddev: 0.0000112865199937",
            "extra": "mean: 46.8542598235 usec\nrounds: 17841"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-compile]",
            "value": 11686.5221206,
            "unit": "iter/sec",
            "range": "stddev: 0.00000487710647041",
            "extra": "mean: 85.568656755 usec\nrounds: 9861"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-eager]",
            "value": 44337.9478119,
            "unit": "iter/sec",
            "range": "stddev: 0.0000112127434056",
            "extra": "mean: 22.5540434177 usec\nrounds: 34806"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-compile]",
            "value": 15207.3257028,
            "unit": "iter/sec",
            "range": "stddev: 0.0000139484488012",
            "extra": "mean: 65.7577814496 usec\nrounds: 13024"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-eager]",
            "value": 47016.3673527,
            "unit": "iter/sec",
            "range": "stddev: 0.0000108916562272",
            "extra": "mean: 21.2691889294 usec\nrounds: 36403"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-compile]",
            "value": 15178.9056655,
            "unit": "iter/sec",
            "range": "stddev: 0.0000046512071293",
            "extra": "mean: 65.8809022231 usec\nrounds: 13465"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-eager]",
            "value": 45938.3212571,
            "unit": "iter/sec",
            "range": "stddev: 0.0000115227407269",
            "extra": "mean: 21.7683183154 usec\nrounds: 36376"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-compile]",
            "value": 11653.4107769,
            "unit": "iter/sec",
            "range": "stddev: 0.00000424656353222",
            "extra": "mean: 85.8117867076 usec\nrounds: 10059"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-eager]",
            "value": 45065.7451918,
            "unit": "iter/sec",
            "range": "stddev: 0.0000108596208825",
            "extra": "mean: 22.1898028257 usec\nrounds: 36089"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-compile]",
            "value": 15549.2668742,
            "unit": "iter/sec",
            "range": "stddev: 0.00000344138030167",
            "extra": "mean: 64.3117137349 usec\nrounds: 13275"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-eager]",
            "value": 45918.6389985,
            "unit": "iter/sec",
            "range": "stddev: 0.0000108172001508",
            "extra": "mean: 21.7776489419 usec\nrounds: 36166"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-compile]",
            "value": 15384.388927,
            "unit": "iter/sec",
            "range": "stddev: 0.00000319240936968",
            "extra": "mean: 65.0009567976 usec\nrounds: 13727"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-eager]",
            "value": 46967.676685,
            "unit": "iter/sec",
            "range": "stddev: 0.0000105649668243",
            "extra": "mean: 21.2912383703 usec\nrounds: 37538"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-eager]",
            "value": 34349.8783489,
            "unit": "iter/sec",
            "range": "stddev: 0.00000199293003121",
            "extra": "mean: 29.1121846152 usec\nrounds: 25119"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-compile]",
            "value": 7011.46015004,
            "unit": "iter/sec",
            "range": "stddev: 0.00000838597993526",
            "extra": "mean: 142.623644519 usec\nrounds: 4261"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-eager]",
            "value": 2718.93249524,
            "unit": "iter/sec",
            "range": "stddev: 0.0000136948947525",
            "extra": "mean: 367.791404071 usec\nrounds: 2594"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-compile]",
            "value": 6524.40892867,
            "unit": "iter/sec",
            "range": "stddev: 0.00000803504357561",
            "extra": "mean: 153.270589096 usec\nrounds: 3415"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[eager]",
            "value": 6052.19394198,
            "unit": "iter/sec",
            "range": "stddev: 0.00000555926497807",
            "extra": "mean: 165.229338251 usec\nrounds: 5330"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[compile]",
            "value": 5814.96808944,
            "unit": "iter/sec",
            "range": "stddev: 0.00000652677716712",
            "extra": "mean: 171.969989279 usec\nrounds: 3827"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[eager]",
            "value": 5710.27443255,
            "unit": "iter/sec",
            "range": "stddev: 0.00000700873079844",
            "extra": "mean: 175.122931798 usec\nrounds: 5584"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[compile]",
            "value": 5375.48954928,
            "unit": "iter/sec",
            "range": "stddev: 0.00000613847515856",
            "extra": "mean: 186.029568253 usec\nrounds: 3612"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-eager]",
            "value": 109568.302119,
            "unit": "iter/sec",
            "range": "stddev: 6.66588349849e-7",
            "extra": "mean: 9.12672717076 usec\nrounds: 84460"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-compile]",
            "value": 44384.6657735,
            "unit": "iter/sec",
            "range": "stddev: 0.00000196830922514",
            "extra": "mean: 22.5303037112 usec\nrounds: 24414"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-eager]",
            "value": 65455.9870381,
            "unit": "iter/sec",
            "range": "stddev: 0.00000126067327276",
            "extra": "mean: 15.2774413046 usec\nrounds: 55553"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-compile]",
            "value": 35958.3978669,
            "unit": "iter/sec",
            "range": "stddev: 0.00000262387972023",
            "extra": "mean: 27.8099153277 usec\nrounds: 23223"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-eager]",
            "value": 37160.0463339,
            "unit": "iter/sec",
            "range": "stddev: 0.00000176224491696",
            "extra": "mean: 26.9106230658 usec\nrounds: 33635"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-compile]",
            "value": 26335.6520709,
            "unit": "iter/sec",
            "range": "stddev: 0.00000278854643129",
            "extra": "mean: 37.9713400416 usec\nrounds: 18087"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[eager]",
            "value": 38391.0937516,
            "unit": "iter/sec",
            "range": "stddev: 0.00000159708269678",
            "extra": "mean: 26.0477080041 usec\nrounds: 29797"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[compile]",
            "value": 8071.62702526,
            "unit": "iter/sec",
            "range": "stddev: 0.00000706224300495",
            "extra": "mean: 123.890759183 usec\nrounds: 7005"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[eager]",
            "value": 32509.3862425,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024826054243",
            "extra": "mean: 30.7603469514 usec\nrounds: 21944"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile]",
            "value": 8969.55926055,
            "unit": "iter/sec",
            "range": "stddev: 0.000016233231098",
            "extra": "mean: 111.488198132 usec\nrounds: 5700"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile-overhead]",
            "value": 5113.69107634,
            "unit": "iter/sec",
            "range": "stddev: 0.0000117834383755",
            "extra": "mean: 195.553463256 usec\nrounds: 302"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[eager]",
            "value": 3873.28689196,
            "unit": "iter/sec",
            "range": "stddev: 0.00000778331918971",
            "extra": "mean: 258.17865495 usec\nrounds: 3336"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile]",
            "value": 3171.56363505,
            "unit": "iter/sec",
            "range": "stddev: 0.00000688475560097",
            "extra": "mean: 315.301887356 usec\nrounds: 2505"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile-overhead]",
            "value": 250.807416792,
            "unit": "iter/sec",
            "range": "stddev: 0.00179827565719",
            "extra": "mean: 3.98712292 msec\nrounds: 1037"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[False]",
            "value": 6364.17463898,
            "unit": "iter/sec",
            "range": "stddev: 0.00000515945116403",
            "extra": "mean: 157.129566162 usec\nrounds: 5212"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[True]",
            "value": 6256.16136142,
            "unit": "iter/sec",
            "range": "stddev: 0.00000702525149704",
            "extra": "mean: 159.842424488 usec\nrounds: 5486"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[eager]",
            "value": 670.397256847,
            "unit": "iter/sec",
            "range": "stddev: 0.0000256548699966",
            "extra": "mean: 1.49165288161 msec\nrounds: 643"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile]",
            "value": 736.731563503,
            "unit": "iter/sec",
            "range": "stddev: 0.000103147491283",
            "extra": "mean: 1.35734648757 msec\nrounds: 664"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile-overhead]",
            "value": 1066.00607833,
            "unit": "iter/sec",
            "range": "stddev: 0.0000769125908464",
            "extra": "mean: 938.080955 usec\nrounds: 88"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[eager]",
            "value": 10335.7101484,
            "unit": "iter/sec",
            "range": "stddev: 0.00000399149099193",
            "extra": "mean: 96.7519392132 usec\nrounds: 8685"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile]",
            "value": 8081.26006057,
            "unit": "iter/sec",
            "range": "stddev: 0.00000461595493075",
            "extra": "mean: 123.74307874 usec\nrounds: 6951"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile-overhead]",
            "value": 4938.36771583,
            "unit": "iter/sec",
            "range": "stddev: 0.00000776973859861",
            "extra": "mean: 202.496058929 usec\nrounds: 3279"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[eager]",
            "value": 2529.97809529,
            "unit": "iter/sec",
            "range": "stddev: 0.00000963738147899",
            "extra": "mean: 395.260339155 usec\nrounds: 2425"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile]",
            "value": 2939.97339065,
            "unit": "iter/sec",
            "range": "stddev: 0.00000911944136799",
            "extra": "mean: 340.139132953 usec\nrounds: 2761"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile-overhead]",
            "value": 3271.69741838,
            "unit": "iter/sec",
            "range": "stddev: 0.00000860417003474",
            "extra": "mean: 305.651737347 usec\nrounds: 2311"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-eager]",
            "value": 1253.07163163,
            "unit": "iter/sec",
            "range": "stddev: 0.0000152017108085",
            "extra": "mean: 798.038974593 usec\nrounds: 1231"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile]",
            "value": 1419.13045864,
            "unit": "iter/sec",
            "range": "stddev: 0.0000185116293843",
            "extra": "mean: 704.6568509 usec\nrounds: 1369"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile-overhead]",
            "value": 2124.4527287,
            "unit": "iter/sec",
            "range": "stddev: 0.0000166815684328",
            "extra": "mean: 470.709461544 usec\nrounds: 1259"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-eager]",
            "value": 785.523459345,
            "unit": "iter/sec",
            "range": "stddev: 0.000021820889399",
            "extra": "mean: 1.273036455 msec\nrounds: 753"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile]",
            "value": 1380.7258407,
            "unit": "iter/sec",
            "range": "stddev: 0.0000157157179567",
            "extra": "mean: 724.256742737 usec\nrounds: 1291"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile-overhead]",
            "value": 2024.14871683,
            "unit": "iter/sec",
            "range": "stddev: 0.0000140522231986",
            "extra": "mean: 494.0348462 usec\nrounds: 1274"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-eager]",
            "value": 1251.96161949,
            "unit": "iter/sec",
            "range": "stddev: 0.0000179955210959",
            "extra": "mean: 798.746530593 usec\nrounds: 1204"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile]",
            "value": 1434.93580064,
            "unit": "iter/sec",
            "range": "stddev: 0.0000165518135794",
            "extra": "mean: 696.895289362 usec\nrounds: 1383"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile-overhead]",
            "value": 2212.17747967,
            "unit": "iter/sec",
            "range": "stddev: 0.0000103335062107",
            "extra": "mean: 452.04329634 usec\nrounds: 1096"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-eager]",
            "value": 759.970666983,
            "unit": "iter/sec",
            "range": "stddev: 0.0000176680254752",
            "extra": "mean: 1.31584025995 msec\nrounds: 749"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile]",
            "value": 1239.11645947,
            "unit": "iter/sec",
            "range": "stddev: 0.0000125405716449",
            "extra": "mean: 807.026645764 usec\nrounds: 1146"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile-overhead]",
            "value": 1345.87429537,
            "unit": "iter/sec",
            "range": "stddev: 0.0000471360001823",
            "extra": "mean: 743.011441293 usec\nrounds: 1056"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[eager]",
            "value": 346.419208808,
            "unit": "iter/sec",
            "range": "stddev: 0.00016016200103",
            "extra": "mean: 2.8866759538 msec\nrounds: 349"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile]",
            "value": 1301.46149081,
            "unit": "iter/sec",
            "range": "stddev: 0.0000226180973076",
            "extra": "mean: 768.366952891 usec\nrounds: 1069"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile-overhead]",
            "value": 1980.80093646,
            "unit": "iter/sec",
            "range": "stddev: 0.0000320349737358",
            "extra": "mean: 504.846287982 usec\nrounds: 128"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1061.53418824,
            "unit": "iter/sec",
            "range": "stddev: 0.0000363912810456",
            "extra": "mean: 942.032777727 usec\nrounds: 990"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1063.45408895,
            "unit": "iter/sec",
            "range": "stddev: 0.000033772574289",
            "extra": "mean: 940.332084278 usec\nrounds: 832"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1158.53335981,
            "unit": "iter/sec",
            "range": "stddev: 0.0000357880141471",
            "extra": "mean: 863.16029792 usec\nrounds: 1132"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1162.73365734,
            "unit": "iter/sec",
            "range": "stddev: 0.0000348727176259",
            "extra": "mean: 860.042189098 usec\nrounds: 1152"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-True]",
            "value": 37.7723583844,
            "unit": "iter/sec",
            "range": "stddev: 0.000450973473499",
            "extra": "mean: 26.4743861059 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-False]",
            "value": 38.3736775381,
            "unit": "iter/sec",
            "range": "stddev: 0.000173216276034",
            "extra": "mean: 26.0595299736 msec\nrounds: 39"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-True]",
            "value": 38.4240250222,
            "unit": "iter/sec",
            "range": "stddev: 0.000292512626587",
            "extra": "mean: 26.0253838431 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-False]",
            "value": 38.3180039065,
            "unit": "iter/sec",
            "range": "stddev: 0.000307812213075",
            "extra": "mean: 26.0973928193 msec\nrounds: 39"
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
        "date": 1792423882033,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-2]",
            "value": 4430.54721953,
            "unit": "iter/sec",
            "range": "stddev: 0.00000922168110923",
            "extra": "mean: 225.705753816 usec\nrounds: 134"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-8]",
            "value": 3536.01435359,
            "unit": "iter/sec",
            "range": "stddev: 0.0000128529952294",
            "extra": "mean: 282.804281885 usec\nrounds: 3090"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-32]",
            "value": 1993.96571653,
            "unit": "iter/sec",
            "range": "stddev: 0.0000152645233716",
            "extra": "mean: 501.513136215 usec\nrounds: 1828"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-2]",
            "value": 4445.0188108,
            "unit": "iter/sec",
            "range": "stddev: 0.00000861429981025",
            "extra": "mean: 224.97092646 usec\nrounds: 3931"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-8]",
            "value": 3506.22446562,
            "unit": "iter/sec",
            "range": "stddev: 0.000010821097216",
            "extra": "mean: 285.207068117 usec\nrounds: 3187"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-32]",
            "value": 1773.56788523,
            "unit": "iter/sec",
            "range": "stddev: 0.0000125072785942",
            "extra": "mean: 563.83519815 usec\nrounds: 1645"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-2]",
            "value": 2209.09336172,
            "unit": "iter/sec",
            "range": "stddev: 0.0000135034290331",
            "extra": "mean: 452.674394541 usec\nrounds: 71"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-8]",
            "value": 591.912485344,
            "unit": "iter/sec",
            "range": "stddev: 0.0000275598667789",
            "extra": "mean: 1.68943893694 msec\nrounds: 571"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-32]",
            "value": 150.717957955,
            "unit": "iter/sec",
            "range": "stddev: 0.000056728648487",
            "extra": "mean: 6.63490942662 msec\nrounds: 150"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-2]",
            "value": 2204.23715149,
            "unit": "iter/sec",
            "range": "stddev: 0.0000144604131924",
            "extra": "mean: 453.671692868 usec\nrounds: 2071"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-8]",
            "value": 588.266450744,
            "unit": "iter/sec",
            "range": "stddev: 0.0000276774034903",
            "extra": "mean: 1.69990996212 msec\nrounds: 581"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-32]",
            "value": 146.734855739,
            "unit": "iter/sec",
            "range": "stddev: 0.0000977837419676",
            "extra": "mean: 6.81501334472 msec\nrounds: 148"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-2]",
            "value": 2962.52491555,
            "unit": "iter/sec",
            "range": "stddev: 0.0000104883644959",
            "extra": "mean: 337.549903716 usec\nrounds: 2244"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-8]",
            "value": 807.436634189,
            "unit": "iter/sec",
            "range": "stddev: 0.0000211960073038",
            "extra": "mean: 1.23848727895 msec\nrounds: 778"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-32]",
            "value": 205.547450708,
            "unit": "iter/sec",
            "range": "stddev: 0.0000554829825853",
            "extra": "mean: 4.86505668914 msec\nrounds: 206"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-2]",
            "value": 2907.16822255,
            "unit": "iter/sec",
            "range": "stddev: 0.0000127091724079",
            "extra": "mean: 343.977342709 usec\nrounds: 2810"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-8]",
            "value": 788.491238272,
            "unit": "iter/sec",
            "range": "stddev: 0.0000231285903085",
            "extra": "mean: 1.26824491061 msec\nrounds: 783"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-32]",
            "value": 204.138719574,
            "unit": "iter/sec",
            "range": "stddev: 0.0000526199624675",
            "extra": "mean: 4.89862972632 msec\nrounds: 201"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-2]",
            "value": 3297.34308402,
            "unit": "iter/sec",
            "range": "stddev: 0.0000124230700581",
            "extra": "mean: 303.274477214 usec\nrounds: 2808"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-8]",
            "value": 895.986120732,
            "unit": "iter/sec",
            "range": "stddev: 0.0000227867433174",
            "extra": "mean: 1.11608871707 msec\nrounds: 887"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-32]",
            "value": 229.762603899,
            "unit": "iter/sec",
            "range": "stddev: 0.0000486615979972",
            "extra": "mean: 4.35231836265 msec\nrounds: 226"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-2]",
            "value": 3306.05095022,
            "unit": "iter/sec",
            "range": "stddev: 0.0000112264883809",
            "extra": "mean: 302.475677192 usec\nrounds: 3023"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-8]",
            "value": 900.73940844,
            "unit": "iter/sec",
            "range": "stddev: 0.0000216510530965",
            "extra": "mean: 1.11019901053 msec\nrounds: 850"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-32]",
            "value": 228.941789942,
            "unit": "iter/sec",
            "range": "stddev: 0.0000443280376767",
            "extra": "mean: 4.36792251976 msec\nrounds: 227"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-128]",
            "value": 187.215148333,
            "unit": "iter/sec",
            "range": "stddev: 0.0000331267447694",
            "extra": "mean: 5.3414481088 msec\nrounds: 184"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-256]",
            "value": 98.5456194048,
            "unit": "iter/sec",
            "range": "stddev: 0.0000786774074957",
            "extra": "mean: 10.1475844998 msec\nrounds: 96"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-128]",
            "value": 180.031760324,
            "unit": "iter/sec",
            "range": "stddev: 0.0000561658684984",
            "extra": "mean: 5.55457547156 msec\nrounds: 176"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-256]",
            "value": 93.6164564289,
            "unit": "iter/sec",
            "range": "stddev: 0.0000770821016018",
            "extra": "mean: 10.681882632 msec\nrounds: 87"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-128]",
            "value": 12.7989300506,
            "unit": "iter/sec",
            "range": "stddev: 0.000261717647963",
            "extra": "mean: 78.1315309986 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-256]",
            "value": 6.42602401955,
            "unit": "iter/sec",
            "range": "stddev: 0.000427332799204",
            "extra": "mean: 155.617221 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-128]",
            "value": 12.5439246509,
            "unit": "iter/sec",
            "range": "stddev: 0.000343670008603",
            "extra": "mean: 79.7198666153 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-256]",
            "value": 6.16561418927,
            "unit": "iter/sec",
            "range": "stddev: 0.000273894688828",
            "extra": "mean: 162.189843429 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-128]",
            "value": 22.9953539496,
            "unit": "iter/sec",
            "range": "stddev: 0.0002022026754",
            "extra": "mean: 43.487045348 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-256]",
            "value": 11.4844229555,
            "unit": "iter/sec",
            "range": "stddev: 0.000314319892829",
            "extra": "mean: 87.0744663337 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-128]",
            "value": 22.6495821095,
            "unit": "iter/sec",
            "range": "stddev: 0.000284956595823",
            "extra": "mean: 44.1509249559 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-256]",
            "value": 11.3592368,
            "unit": "iter/sec",
            "range": "stddev: 0.000224800799727",
            "extra": "mean: 88.0340834165 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-128]",
            "value": 25.6494227662,
            "unit": "iter/sec",
            "range": "stddev: 0.000181612381274",
            "extra": "mean: 38.9872321539 msec\nrounds: 26"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-256]",
            "value": 12.6964621994,
            "unit": "iter/sec",
            "range": "stddev: 0.000752644154039",
            "extra": "mean: 78.762098 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-128]",
            "value": 25.3818039557,
            "unit": "iter/sec",
            "range": "stddev: 0.000424994687011",
            "extra": "mean: 39.3983028845 msec\nrounds: 26"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-256]",
            "value": 12.7499232923,
            "unit": "iter/sec",
            "range": "stddev: 0.00295641075473",
            "extra": "mean: 78.4318444175 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_mask_many_leaves[cuda]",
            "value": 6089.59886996,
            "unit": "iter/sec",
            "range": "stddev: 0.0000185429043563",
            "extra": "mean: 164.214428791 usec\nrounds: 42"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-None]",
            "value": 6463.33051048,
            "unit": "iter/sec",
            "range": "stddev: 0.00000711456406434",
            "extra": "mean: 154.718994855 usec\nrounds: 1382"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-cpu]",
            "value": 6510.97599431,
            "unit": "iter/sec",
            "range": "stddev: 0.00000701271083575",
            "extra": "mean: 153.586804939 usec\nrounds: 5147"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to",
            "value": 22360.9995703,
            "unit": "iter/sec",
            "range": "stddev: 0.00000357460475845",
            "extra": "mean: 44.7207199686 usec\nrounds: 11367"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_nonblocking",
            "value": 22483.0173965,
            "unit": "iter/sec",
            "range": "stddev: 0.00000340428729141",
            "extra": "mean: 44.4780156669 usec\nrounds: 12191"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[add]",
            "value": 10320.6305751,
            "unit": "iter/sec",
            "range": "stddev: 0.00000834028603115",
            "extra": "mean: 96.8933044083 usec\nrounds: 174"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[sub]",
            "value": 10483.9012688,
            "unit": "iter/sec",
            "range": "stddev: 0.00000555617557131",
            "extra": "mean: 95.3843396991 usec\nrounds: 7922"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_blocking",
            "value": 34.1563317198,
            "unit": "iter/sec",
            "range": "stddev: 0.000392748319355",
            "extra": "mean: 29.2771486178 msec\nrounds: 34"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_global_sync",
            "value": 93.8635326338,
            "unit": "iter/sec",
            "range": "stddev: 0.0125092980992",
            "extra": "mean: 10.6537648002 msec\nrounds: 105"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_event_sync",
            "value": 95.9878902482,
            "unit": "iter/sec",
            "range": "stddev: 0.000126386460701",
            "extra": "mean: 10.4179808246 msec\nrounds: 97"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_default",
            "value": 93.8547320229,
            "unit": "iter/sec",
            "range": "stddev: 0.000112782911554",
            "extra": "mean: 10.654763787 msec\nrounds: 94"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_cpu_to_cpu_default",
            "value": 3876.59715014,
            "unit": "iter/sec",
            "range": "stddev: 0.00000992090861018",
            "extra": "mean: 257.958194073 usec\nrounds: 3818"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[False-None]",
            "value": 288.389959791,
            "unit": "iter/sec",
            "range": "stddev: 0.000104857154574",
            "extra": "mean: 3.46752709673 msec\nrounds: 289"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[default-None]",
            "value": 413.472442707,
            "unit": "iter/sec",
            "range": "stddev: 0.000225628629607",
            "extra": "mean: 2.41854086684 msec\nrounds: 420"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[reduce-overhead-None]",
            "value": 437.462256139,
            "unit": "iter/sec",
            "range": "stddev: 0.000279344219802",
            "extra": "mean: 2.28591149514 msec\nrounds: 414"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[False-False-None]",
            "value": 461.096169743,
            "unit": "iter/sec",
            "range": "stddev: 0.0000322534775814",
            "extra": "mean: 2.16874497256 msec\nrounds: 439"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-False-None]",
            "value": 610.003170791,
            "unit": "iter/sec",
            "range": "stddev: 0.0000639677363914",
            "extra": "mean: 1.639335741 msec\nrounds: 610"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[within-False-None]",
            "value": 187.296573804,
            "unit": "iter/sec",
            "range": "stddev: 0.0000599750598735",
            "extra": "mean: 5.33912596312 msec\nrounds: 190"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-default-None]",
            "value": 107.527371845,
            "unit": "iter/sec",
            "range": "stddev: 0.0000678344375463",
            "extra": "mean: 9.29995760931 msec\nrounds: 105"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[False-False-None]",
            "value": 133.329925659,
            "unit": "iter/sec",
            "range": "stddev: 0.0000588857632417",
            "extra": "mean: 7.5001916866 msec\nrounds: 134"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[True-False-None]",
            "value": 248.758549126,
            "unit": "iter/sec",
            "range": "stddev: 0.0000478357556556",
            "extra": "mean: 4.01996234306 msec\nrounds: 239"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[within-False-None]",
            "value": 104.202146185,
            "unit": "iter/sec",
            "range": "stddev: 0.000070310368694",
            "extra": "mean: 9.59673132093 msec\nrounds: 106"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 14.903956327,
            "unit": "iter/sec",
            "range": "stddev: 0.000575620925764",
            "extra": "mean: 67.0962782002 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_archive",
            "value": 4.40010814903,
            "unit": "iter/sec",
            "range": "stddev: 0.00225197682775",
            "extra": "mean: 227.2671412 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_zarr",
            "value": 0.635331186095,
            "unit": "iter/sec",
            "range": "stddev: 0.102391875432",
            "extra": "mean: 1.5739822346 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 37.6016585273,
            "unit": "iter/sec",
            "range": "stddev: 0.000466884523392",
            "extra": "mean: 26.5945716004 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 15.1512711704,
            "unit": "iter/sec",
            "range": "stddev: 0.000129342511966",
            "extra": "mean: 66.0010627988 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 15.4534778411,
            "unit": "iter/sec",
            "range": "stddev: 0.000822853696371",
            "extra": "mean: 64.7103526004 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 39.8384912518,
            "unit": "iter/sec",
            "range": "stddev: 0.000336319119979",
            "extra": "mean: 25.1013521993 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-compile]",
            "value": 4582.35831551,
            "unit": "iter/sec",
            "range": "stddev: 0.000011662013655",
            "extra": "mean: 218.228242129 usec\nrounds: 1623"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-eager]",
            "value": 7839.271351,
            "unit": "iter/sec",
            "range": "stddev: 0.00000801243280926",
            "extra": "mean: 127.562876093 usec\nrounds: 6699"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-compile]",
            "value": 6167.33831699,
            "unit": "iter/sec",
            "range": "stddev: 0.0000142000797745",
            "extra": "mean: 162.144501988 usec\nrounds: 2751"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-eager]",
            "value": 5548.99511857,
            "unit": "iter/sec",
            "range": "stddev: 0.0000208992780176",
            "extra": "mean: 180.21280946 usec\nrounds: 5175"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-compile]",
            "value": 19545.0694653,
            "unit": "iter/sec",
            "range": "stddev: 0.00000358648480648",
            "extra": "mean: 51.1637987154 usec\nrounds: 15593"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-eager]",
            "value": 30686.1483666,
            "unit": "iter/sec",
            "range": "stddev: 0.00000207082563454",
            "extra": "mean: 32.5879933856 usec\nrounds: 24194"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-compile]",
            "value": 48446.5852477,
            "unit": "iter/sec",
            "range": "stddev: 0.00000397074057429",
            "extra": "mean: 20.6412896778 usec\nrounds: 33978"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-eager]",
            "value": 15696.1896946,
            "unit": "iter/sec",
            "range": "stddev: 0.0000116307491291",
            "extra": "mean: 63.7097295239 usec\nrounds: 14793"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-compile]",
            "value": 3587.20426172,
            "unit": "iter/sec",
            "range": "stddev: 0.00000903356296931",
            "extra": "mean: 278.768625102 usec\nrounds: 1059"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-eager]",
            "value": 5120.99446195,
            "unit": "iter/sec",
            "range": "stddev: 0.00000906093685548",
            "extra": "mean: 195.274571654 usec\nrounds: 4891"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-compile]",
            "value": 5467.09970673,
            "unit": "iter/sec",
            "range": "stddev: 0.0000115090334219",
            "extra": "mean: 182.912339932 usec\nrounds: 3192"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-eager]",
            "value": 15661.0091353,
            "unit": "iter/sec",
            "range": "stddev: 0.00000464293288973",
            "extra": "mean: 63.8528457115 usec\nrounds: 13349"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-compile]",
            "value": 3953.95878221,
            "unit": "iter/sec",
            "range": "stddev: 0.0000111972168783",
            "extra": "mean: 252.911083569 usec\nrounds: 1364"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-eager]",
            "value": 1921.48231467,
            "unit": "iter/sec",
            "range": "stddev: 0.0000216838526231",
            "extra": "mean: 520.431539945 usec\nrounds: 1906"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-eager]",
            "value": 4271.21705336,
            "unit": "iter/sec",
            "range": "stddev: 0.0000100037786852",
            "extra": "mean: 234.125306091 usec\nrounds: 4048"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-compile]",
            "value": 3566.66933145,
            "unit": "iter/sec",
            "range": "stddev: 0.0000092356699113",
            "extra": "mean: 280.373622299 usec\nrounds: 1316"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-eager]",
            "value": 12730.7757359,
            "unit": "iter/sec",
            "range": "stddev: 0.00000487002947369",
            "extra": "mean: 78.5498088052 usec\nrounds: 11014"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-compile]",
            "value": 5592.45636016,
            "unit": "iter/sec",
            "range": "stddev: 0.00000808741664489",
            "extra": "mean: 178.812302788 usec\nrounds: 3161"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-eager]",
            "value": 2193.35091531,
            "unit": "iter/sec",
            "range": "stddev: 0.000019452313793",
            "extra": "mean: 455.923396945 usec\nrounds: 2174"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-compile]",
            "value": 4007.38220817,
            "unit": "iter/sec",
            "range": "stddev: 0.000010059992781",
            "extra": "mean: 249.539461937 usec\nrounds: 1366"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-compile]",
            "value": 30556.3416532,
            "unit": "iter/sec",
            "range": "stddev: 0.00000296891253371",
            "extra": "mean: 32.7264307799 usec\nrounds: 21087"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-eager]",
            "value": 54562.8110805,
            "unit": "iter/sec",
            "range": "stddev: 0.00000126668205255",
            "extra": "mean: 18.3275014648 usec\nrounds: 46793"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-compile]",
            "value": 42458.4820456,
            "unit": "iter/sec",
            "range": "stddev: 0.0000038445487305",
            "extra": "mean: 23.5524199599 usec\nrounds: 26637"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-eager]",
            "value": 19486.1818,
            "unit": "iter/sec",
            "range": "stddev: 0.0000116568182567",
            "extra": "mean: 51.318416828 usec\nrounds: 16995"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-compile]",
            "value": 3795.51717108,
            "unit": "iter/sec",
            "range": "stddev: 0.000227564189538",
            "extra": "mean: 263.468706615 usec\nrounds: 92"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-eager]",
            "value": 292.020787341,
            "unit": "iter/sec",
            "range": "stddev: 0.0000419370998389",
            "extra": "mean: 3.42441375186 msec\nrounds: 290"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-compile]",
            "value": 4856.8315151,
            "unit": "iter/sec",
            "range": "stddev: 0.000223826176269",
            "extra": "mean: 205.895550812 usec\nrounds: 89"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-eager]",
            "value": 343.886246218,
            "unit": "iter/sec",
            "range": "stddev: 0.000045850843714",
            "extra": "mean: 2.90793834007 msec\nrounds: 347"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-compile]",
            "value": 5926.25209927,
            "unit": "iter/sec",
            "range": "stddev: 0.00000969640098595",
            "extra": "mean: 168.740712215 usec\nrounds: 3791"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-eager]",
            "value": 17967.716832,
            "unit": "iter/sec",
            "range": "stddev: 0.00001291943044",
            "extra": "mean: 55.6553739882 usec\nrounds: 14607"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-compile]",
            "value": 7312.21011077,
            "unit": "iter/sec",
            "range": "stddev: 0.00000764015660999",
            "extra": "mean: 136.757558228 usec\nrounds: 4373"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-eager]",
            "value": 20819.5529305,
            "unit": "iter/sec",
            "range": "stddev: 0.0000125239592649",
            "extra": "mean: 48.0317710634 usec\nrounds: 17730"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-compile]",
            "value": 7242.20312972,
            "unit": "iter/sec",
            "range": "stddev: 0.00000794566663661",
            "extra": "mean: 138.079529404 usec\nrounds: 6517"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-eager]",
            "value": 20995.7023625,
            "unit": "iter/sec",
            "range": "stddev: 0.0000120802421898",
            "extra": "mean: 47.6287948236 usec\nrounds: 17771"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-compile]",
            "value": 11217.5578065,
            "unit": "iter/sec",
            "range": "stddev: 0.0000058502746777",
            "extra": "mean: 89.1459636091 usec\nrounds: 9338"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-eager]",
            "value": 43731.2181156,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113326252801",
            "extra": "mean: 22.8669596479 usec\nrounds: 33522"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-compile]",
            "value": 15230.4003264,
            "unit": "iter/sec",
            "range": "stddev: 0.0000128125332902",
            "extra": "mean: 65.6581559625 usec\nrounds: 12189"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-eager]",
            "value": 44172.6219958,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113711895291",
            "extra": "mean: 22.6384569179 usec\nrounds: 34829"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-compile]",
            "value": 14843.452409,
            "unit": "iter/sec",
            "range": "stddev: 0.00000446337712117",
            "extra": "mean: 67.3697716978 usec\nrounds: 13009"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-eager]",
            "value": 43850.0701018,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113093404574",
            "extra": "mean: 22.8049806461 usec\nrounds: 35675"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-compile]",
            "value": 11117.5908759,
            "unit": "iter/sec",
            "range": "stddev: 0.00000583082509217",
            "extra": "mean: 89.9475444961 usec\nrounds: 10002"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-eager]",
            "value": 43100.4443703,
            "unit": "iter/sec",
            "range": "stddev: 0.0000108774592821",
            "extra": "mean: 23.2016169348 usec\nrounds: 33817"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-compile]",
            "value": 14762.3584994,
            "unit": "iter/sec",
            "range": "stddev: 0.00000434605873322",
            "extra": "mean: 67.7398533601 usec\nrounds: 12731"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-eager]",
            "value": 44805.7413183,
            "unit": "iter/sec",
            "range": "stddev: 0.0000110441930525",
            "extra": "mean: 22.3185683481 usec\nrounds: 34129"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-compile]",
            "value": 14855.8993641,
            "unit": "iter/sec",
            "range": "stddev: 0.00000415004784666",
            "extra": "mean: 67.3133262075 usec\nrounds: 13121"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-eager]",
            "value": 44329.978233,
            "unit": "iter/sec",
            "range": "stddev: 0.000010993235824",
            "extra": "mean: 22.5580981508 usec\nrounds: 35854"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-eager]",
            "value": 31707.1979794,
            "unit": "iter/sec",
            "range": "stddev: 0.00000284332482179",
            "extra": "mean: 31.5385799985 usec\nrounds: 22924"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-compile]",
            "value": 6679.96820701,
            "unit": "iter/sec",
            "range": "stddev: 0.0000119598120101",
            "extra": "mean: 149.701311295 usec\nrounds: 4247"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-eager]",
            "value": 2495.21100719,
            "unit": "iter/sec",
            "range": "stddev: 0.0000176256694381",
            "extra": "mean: 400.767709471 usec\nrounds: 2350"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-compile]",
            "value": 6205.90525251,
            "unit": "iter/sec",
            "range": "stddev: 0.00000900289571538",
            "extra": "mean: 161.136846167 usec\nrounds: 3387"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[eager]",
            "value": 5884.21463588,
            "unit": "iter/sec",
            "range": "stddev: 0.00000928878622744",
            "extra": "mean: 169.946214046 usec\nrounds: 5658"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[compile]",
            "value": 5239.22978301,
            "unit": "iter/sec",
            "range": "stddev: 0.00001009833027",
            "extra": "mean: 190.867749921 usec\nrounds: 3451"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[eager]",
            "value": 5745.38001007,
            "unit": "iter/sec",
            "range": "stddev: 0.00000850753631001",
            "extra": "mean: 174.052890888 usec\nrounds: 5517"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[compile]",
            "value": 4880.37577962,
            "unit": "iter/sec",
            "range": "stddev: 0.0000100680623386",
            "extra": "mean: 204.90225449 usec\nrounds: 3517"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-eager]",
            "value": 107257.810774,
            "unit": "iter/sec",
            "range": "stddev: 8.78022919224e-7",
            "extra": "mean: 9.323330327 usec\nrounds: 81894"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-compile]",
            "value": 36883.0070528,
            "unit": "iter/sec",
            "range": "stddev: 0.00000484486064824",
            "extra": "mean: 27.1127567925 usec\nrounds: 23923"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-eager]",
            "value": 64801.5460309,
            "unit": "iter/sec",
            "range": "stddev: 0.00000117015578861",
            "extra": "mean: 15.4317305875 usec\nrounds: 46684"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-compile]",
            "value": 30336.4609568,
            "unit": "iter/sec",
            "range": "stddev: 0.00000266520562855",
            "extra": "mean: 32.9636341373 usec\nrounds: 20669"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-eager]",
            "value": 36394.2607007,
            "unit": "iter/sec",
            "range": "stddev: 0.00000174396518289",
            "extra": "mean: 27.476859833 usec\nrounds: 33167"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-compile]",
            "value": 22576.5297908,
            "unit": "iter/sec",
            "range": "stddev: 0.00000397886766057",
            "extra": "mean: 44.2937869223 usec\nrounds: 16961"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[eager]",
            "value": 35859.2792503,
            "unit": "iter/sec",
            "range": "stddev: 0.00000253560940034",
            "extra": "mean: 27.8867847014 usec\nrounds: 27502"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[compile]",
            "value": 8010.021742,
            "unit": "iter/sec",
            "range": "stddev: 0.00000672391086165",
            "extra": "mean: 124.843606199 usec\nrounds: 6968"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[eager]",
            "value": 29517.5536922,
            "unit": "iter/sec",
            "range": "stddev: 0.00000266227854422",
            "extra": "mean: 33.8781462186 usec\nrounds: 21303"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile]",
            "value": 8690.31667031,
            "unit": "iter/sec",
            "range": "stddev: 0.0000325431840531",
            "extra": "mean: 115.070605357 usec\nrounds: 5666"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile-overhead]",
            "value": 4964.92693842,
            "unit": "iter/sec",
            "range": "stddev: 0.0000158485895005",
            "extra": "mean: 201.412832939 usec\nrounds: 299"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[eager]",
            "value": 3590.10333558,
            "unit": "iter/sec",
            "range": "stddev: 0.0000134424037829",
            "extra": "mean: 278.543514358 usec\nrounds: 3276"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile]",
            "value": 3007.19114808,
            "unit": "iter/sec",
            "range": "stddev: 0.0000120998216582",
            "extra": "mean: 332.536227582 usec\nrounds: 2491"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile-overhead]",
            "value": 255.400413469,
            "unit": "iter/sec",
            "range": "stddev: 0.00173499000512",
            "extra": "mean: 3.91542044281 msec\nrounds: 987"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[False]",
            "value": 6006.24527939,
            "unit": "iter/sec",
            "range": "stddev: 0.00000760366018725",
            "extra": "mean: 166.493367068 usec\nrounds: 4980"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[True]",
            "value": 5939.44196847,
            "unit": "iter/sec",
            "range": "stddev: 0.00000826334736049",
            "extra": "mean: 168.365985443 usec\nrounds: 5162"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[eager]",
            "value": 633.152686175,
            "unit": "iter/sec",
            "range": "stddev: 0.000121969091914",
            "extra": "mean: 1.57939786379 msec\nrounds: 631"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile]",
            "value": 748.701503741,
            "unit": "iter/sec",
            "range": "stddev: 0.0000296784088344",
            "extra": "mean: 1.33564577472 msec\nrounds: 657"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile-overhead]",
            "value": 1039.66568795,
            "unit": "iter/sec",
            "range": "stddev: 0.000251734780825",
            "extra": "mean: 961.847651207 usec\nrounds: 86"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[eager]",
            "value": 9832.73257217,
            "unit": "iter/sec",
            "range": "stddev: 0.00000695583765142",
            "extra": "mean: 101.701128619 usec\nrounds: 8490"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile]",
            "value": 7446.67594974,
            "unit": "iter/sec",
            "range": "stddev: 0.00000740569562519",
            "extra": "mean: 134.288104753 usec\nrounds: 6502"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile-overhead]",
            "value": 4645.46467366,
            "unit": "iter/sec",
            "range": "stddev: 0.00000783199900641",
            "extra": "mean: 215.263718541 usec\nrounds: 3162"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[eager]",
            "value": 2371.05552665,
            "unit": "iter/sec",
            "range": "stddev: 0.000013932038822",
            "extra": "mean: 421.75309214 usec\nrounds: 2256"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile]",
            "value": 2747.06406665,
            "unit": "iter/sec",
            "range": "stddev: 0.0000119973080879",
            "extra": "mean: 364.025001142 usec\nrounds: 2616"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile-overhead]",
            "value": 3118.69400002,
            "unit": "iter/sec",
            "range": "stddev: 0.0000121797562331",
            "extra": "mean: 320.647040073 usec\nrounds: 2198"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-eager]",
            "value": 1134.81410475,
            "unit": "iter/sec",
            "range": "stddev: 0.0000191080256752",
            "extra": "mean: 881.201595767 usec\nrounds: 1133"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile]",
            "value": 1327.73655263,
            "unit": "iter/sec",
            "range": "stddev: 0.0000182599939492",
            "extra": "mean: 753.161459645 usec\nrounds: 1275"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile-overhead]",
            "value": 2047.28188557,
            "unit": "iter/sec",
            "range": "stddev: 0.0000151276811361",
            "extra": "mean: 488.452521878 usec\nrounds: 1278"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-eager]",
            "value": 741.517609009,
            "unit": "iter/sec",
            "range": "stddev: 0.0000254307187345",
            "extra": "mean: 1.34858564092 msec\nrounds: 710"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile]",
            "value": 1305.80950222,
            "unit": "iter/sec",
            "range": "stddev: 0.0000212614222149",
            "extra": "mean: 765.808487608 usec\nrounds: 1296"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile-overhead]",
            "value": 1937.68379552,
            "unit": "iter/sec",
            "range": "stddev: 0.000017208863292",
            "extra": "mean: 516.080075763 usec\nrounds: 1161"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-eager]",
            "value": 1157.16459423,
            "unit": "iter/sec",
            "range": "stddev: 0.000020524844355",
            "extra": "mean: 864.1812971 usec\nrounds: 1155"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile]",
            "value": 1335.76330352,
            "unit": "iter/sec",
            "range": "stddev: 0.0000214193025845",
            "extra": "mean: 748.635628305 usec\nrounds: 1232"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile-overhead]",
            "value": 2124.79905908,
            "unit": "iter/sec",
            "range": "stddev: 0.0000128679803046",
            "extra": "mean: 470.63273853 usec\nrounds: 1098"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-eager]",
            "value": 711.492224732,
            "unit": "iter/sec",
            "range": "stddev: 0.0000301581035009",
            "extra": "mean: 1.40549673663 msec\nrounds: 714"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile]",
            "value": 1188.7588721,
            "unit": "iter/sec",
            "range": "stddev: 0.0000181851820509",
            "extra": "mean: 841.213490362 usec\nrounds: 1138"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile-overhead]",
            "value": 1295.64525948,
            "unit": "iter/sec",
            "range": "stddev: 0.000083821104955",
            "extra": "mean: 771.816199447 usec\nrounds: 948"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[eager]",
            "value": 334.454971249,
            "unit": "iter/sec",
            "range": "stddev: 0.0000364751466786",
            "extra": "mean: 2.98993911278 msec\nrounds: 328"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile]",
            "value": 1247.79843864,
            "unit": "iter/sec",
            "range": "stddev: 0.0000153537365153",
            "extra": "mean: 801.411485246 usec\nrounds: 983"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile-overhead]",
            "value": 1884.35363551,
            "unit": "iter/sec",
            "range": "stddev: 0.0000340643501248",
            "extra": "mean: 530.685950425 usec\nrounds: 121"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1003.87620283,
            "unit": "iter/sec",
            "range": "stddev: 0.0000374400353959",
            "extra": "mean: 996.1387641 usec\nrounds: 941"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1017.00725717,
            "unit": "iter/sec",
            "range": "stddev: 0.0000358432148858",
            "extra": "mean: 983.277152594 usec\nrounds: 1009"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1105.54822439,
            "unit": "iter/sec",
            "range": "stddev: 0.0000353402206796",
            "extra": "mean: 904.528611182 usec\nrounds: 1088"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1107.66633074,
            "unit": "iter/sec",
            "range": "stddev: 0.0000358860193689",
            "extra": "mean: 902.798949694 usec\nrounds: 1114"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-True]",
            "value": 36.5605105624,
            "unit": "iter/sec",
            "range": "stddev: 0.0000915043515832",
            "extra": "mean: 27.351915622 msec\nrounds: 37"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-False]",
            "value": 36.6483765651,
            "unit": "iter/sec",
            "range": "stddev: 0.0000715451850914",
            "extra": "mean: 27.28633827 msec\nrounds: 37"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-True]",
            "value": 36.633285644,
            "unit": "iter/sec",
            "range": "stddev: 0.000182315111617",
            "extra": "mean: 27.2975787572 msec\nrounds: 37"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-False]",
            "value": 36.8007987484,
            "unit": "iter/sec",
            "range": "stddev: 0.0000937479432291",
            "extra": "mean: 27.1733232432 msec\nrounds: 37"
          }
        ]
      }
    ]
  }
}