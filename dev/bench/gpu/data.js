window.BENCHMARK_DATA = {
  "lastUpdate": 1792423589661,
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
      }
    ]
  }
}