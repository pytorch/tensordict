window.BENCHMARK_DATA = {
  "lastUpdate": 1792514292705,
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
          "id": "977215952c51209e92bc7449ea41cb5e5e6b16ba",
          "message": "[CI] Run the core unit tests nightly on Windows and Linux ARM, without blocking the upload (#3059)",
          "timestamp": "2026-10-19T11:15:11-04:00",
          "tree_id": "4385c4802410bc665eff3c7ce0d94add2431963c",
          "url": "https://github.com/pytorch/tensordict/commit/977215952c51209e92bc7449ea41cb5e5e6b16ba"
        },
        "date": 1792424027313,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-2]",
            "value": 4556.49972302,
            "unit": "iter/sec",
            "range": "stddev: 0.00000868341464326",
            "extra": "mean: 219.46670927 usec\nrounds: 141"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-8]",
            "value": 3607.46824999,
            "unit": "iter/sec",
            "range": "stddev: 0.00000842872911689",
            "extra": "mean: 277.202716893 usec\nrounds: 3136"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-32]",
            "value": 2031.85162747,
            "unit": "iter/sec",
            "range": "stddev: 0.0000112999379036",
            "extra": "mean: 492.161920921 usec\nrounds: 1822"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-2]",
            "value": 4388.57821064,
            "unit": "iter/sec",
            "range": "stddev: 0.00000761195733804",
            "extra": "mean: 227.864231194 usec\nrounds: 3802"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-8]",
            "value": 3487.3568611,
            "unit": "iter/sec",
            "range": "stddev: 0.00000944620318678",
            "extra": "mean: 286.750120458 usec\nrounds: 3129"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-32]",
            "value": 1795.67003103,
            "unit": "iter/sec",
            "range": "stddev: 0.0000101598497067",
            "extra": "mean: 556.89518827 usec\nrounds: 1678"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-2]",
            "value": 2278.9257453,
            "unit": "iter/sec",
            "range": "stddev: 0.0000110513508754",
            "extra": "mean: 438.803239667 usec\nrounds: 75"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-8]",
            "value": 606.121646324,
            "unit": "iter/sec",
            "range": "stddev: 0.0000418686119706",
            "extra": "mean: 1.64983383462 msec\nrounds: 569"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-32]",
            "value": 153.772757768,
            "unit": "iter/sec",
            "range": "stddev: 0.000120602835097",
            "extra": "mean: 6.50310246443 msec\nrounds: 155"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-2]",
            "value": 2267.05216702,
            "unit": "iter/sec",
            "range": "stddev: 0.0000136722808685",
            "extra": "mean: 441.101450839 usec\nrounds: 2160"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-8]",
            "value": 604.450061432,
            "unit": "iter/sec",
            "range": "stddev: 0.0000349891116201",
            "extra": "mean: 1.65439639071 msec\nrounds: 599"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-32]",
            "value": 151.710451607,
            "unit": "iter/sec",
            "range": "stddev: 0.000112692909537",
            "extra": "mean: 6.59150367959 msec\nrounds: 153"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-2]",
            "value": 3042.74202221,
            "unit": "iter/sec",
            "range": "stddev: 0.0000256140480267",
            "extra": "mean: 328.650931529 usec\nrounds: 2393"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-8]",
            "value": 821.277629748,
            "unit": "iter/sec",
            "range": "stddev: 0.0000563344619083",
            "extra": "mean: 1.21761504731 msec\nrounds: 783"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-32]",
            "value": 211.619126659,
            "unit": "iter/sec",
            "range": "stddev: 0.0000740136013388",
            "extra": "mean: 4.72547078228 msec\nrounds: 193"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-2]",
            "value": 3060.12867766,
            "unit": "iter/sec",
            "range": "stddev: 0.0000182222330149",
            "extra": "mean: 326.783643871 usec\nrounds: 2869"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-8]",
            "value": 833.324973644,
            "unit": "iter/sec",
            "range": "stddev: 0.000027248358578",
            "extra": "mean: 1.20001203807 msec\nrounds: 816"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-32]",
            "value": 213.324353206,
            "unit": "iter/sec",
            "range": "stddev: 0.0000319512857296",
            "extra": "mean: 4.68769732555 msec\nrounds: 209"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-2]",
            "value": 3637.8497336,
            "unit": "iter/sec",
            "range": "stddev: 0.00000832554035742",
            "extra": "mean: 274.887659807 usec\nrounds: 3060"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-8]",
            "value": 963.229925674,
            "unit": "iter/sec",
            "range": "stddev: 0.00003847376811",
            "extra": "mean: 1.03817372503 msec\nrounds: 953"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-32]",
            "value": 245.520497638,
            "unit": "iter/sec",
            "range": "stddev: 0.0000508179388217",
            "extra": "mean: 4.07297968855 msec\nrounds: 247"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-2]",
            "value": 3568.39506962,
            "unit": "iter/sec",
            "range": "stddev: 0.0000176278102659",
            "extra": "mean: 280.238028719 usec\nrounds: 3342"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-8]",
            "value": 962.751318282,
            "unit": "iter/sec",
            "range": "stddev: 0.0000438084569925",
            "extra": "mean: 1.03868982676 msec\nrounds: 952"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-32]",
            "value": 246.267182409,
            "unit": "iter/sec",
            "range": "stddev: 0.0000502327997803",
            "extra": "mean: 4.0606303699 msec\nrounds: 246"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-128]",
            "value": 184.326055483,
            "unit": "iter/sec",
            "range": "stddev: 0.000275027885633",
            "extra": "mean: 5.4251689886 msec\nrounds: 178"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-256]",
            "value": 99.0264571794,
            "unit": "iter/sec",
            "range": "stddev: 0.000061788348032",
            "extra": "mean: 10.0983113855 msec\nrounds: 96"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-128]",
            "value": 181.2132313,
            "unit": "iter/sec",
            "range": "stddev: 0.0000930579038782",
            "extra": "mean: 5.51836084387 msec\nrounds: 179"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-256]",
            "value": 94.1543423827,
            "unit": "iter/sec",
            "range": "stddev: 0.0000238530035992",
            "extra": "mean: 10.6208590565 msec\nrounds: 88"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-128]",
            "value": 12.9365025612,
            "unit": "iter/sec",
            "range": "stddev: 0.000530387722648",
            "extra": "mean: 77.3006456164 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-256]",
            "value": 6.50778004131,
            "unit": "iter/sec",
            "range": "stddev: 0.00113531365507",
            "extra": "mean: 153.662230999 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-128]",
            "value": 12.8259509729,
            "unit": "iter/sec",
            "range": "stddev: 0.000398287094903",
            "extra": "mean: 77.9669283091 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-256]",
            "value": 6.4548693343,
            "unit": "iter/sec",
            "range": "stddev: 0.00112946740804",
            "extra": "mean: 154.921803713 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-128]",
            "value": 23.52234171,
            "unit": "iter/sec",
            "range": "stddev: 0.000166964641731",
            "extra": "mean: 42.5127741246 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-256]",
            "value": 11.8425637611,
            "unit": "iter/sec",
            "range": "stddev: 0.000164021523917",
            "extra": "mean: 84.4411750847 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-128]",
            "value": 23.4253093687,
            "unit": "iter/sec",
            "range": "stddev: 0.000098202692875",
            "extra": "mean: 42.6888705827 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-256]",
            "value": 11.6987126189,
            "unit": "iter/sec",
            "range": "stddev: 0.000268519181718",
            "extra": "mean: 85.4794909986 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-128]",
            "value": 27.4679120979,
            "unit": "iter/sec",
            "range": "stddev: 0.000167685704432",
            "extra": "mean: 36.4061162142 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-256]",
            "value": 13.7281800846,
            "unit": "iter/sec",
            "range": "stddev: 0.000304471691882",
            "extra": "mean: 72.8428672872 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-128]",
            "value": 27.2578926959,
            "unit": "iter/sec",
            "range": "stddev: 0.000416590836787",
            "extra": "mean: 36.6866217854 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-256]",
            "value": 13.5611750748,
            "unit": "iter/sec",
            "range": "stddev: 0.000348118742423",
            "extra": "mean: 73.7399225717 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_mask_many_leaves[cuda]",
            "value": 6441.15908167,
            "unit": "iter/sec",
            "range": "stddev: 0.0000163232833481",
            "extra": "mean: 155.251560677 usec\nrounds: 41"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-None]",
            "value": 6766.82613996,
            "unit": "iter/sec",
            "range": "stddev: 0.00000573723489648",
            "extra": "mean: 147.779768435 usec\nrounds: 1429"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-cpu]",
            "value": 6907.07427104,
            "unit": "iter/sec",
            "range": "stddev: 0.00000543807325879",
            "extra": "mean: 144.779100493 usec\nrounds: 5126"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to",
            "value": 23277.4610277,
            "unit": "iter/sec",
            "range": "stddev: 0.00000249305797297",
            "extra": "mean: 42.9600117817 usec\nrounds: 12032"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_nonblocking",
            "value": 24243.2849749,
            "unit": "iter/sec",
            "range": "stddev: 0.00000247885201398",
            "extra": "mean: 41.2485354619 usec\nrounds: 12519"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[add]",
            "value": 10572.5231792,
            "unit": "iter/sec",
            "range": "stddev: 0.00000825778062147",
            "extra": "mean: 94.5848009079 usec\nrounds: 176"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[sub]",
            "value": 10916.385152,
            "unit": "iter/sec",
            "range": "stddev: 0.00000383735120207",
            "extra": "mean: 91.605415719 usec\nrounds: 8123"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_blocking",
            "value": 56.6077826183,
            "unit": "iter/sec",
            "range": "stddev: 0.0000900997952979",
            "extra": "mean: 17.6654154914 msec\nrounds: 57"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_global_sync",
            "value": 95.7291358155,
            "unit": "iter/sec",
            "range": "stddev: 0.0118588494837",
            "extra": "mean: 10.4461404721 msec\nrounds: 108"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_event_sync",
            "value": 97.3623759438,
            "unit": "iter/sec",
            "range": "stddev: 0.0000852953661643",
            "extra": "mean: 10.2709079386 msec\nrounds: 98"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_default",
            "value": 95.638560699,
            "unit": "iter/sec",
            "range": "stddev: 0.0000967333794907",
            "extra": "mean: 10.4560335569 msec\nrounds: 97"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_cpu_to_cpu_default",
            "value": 3822.12977939,
            "unit": "iter/sec",
            "range": "stddev: 0.00000758514596189",
            "extra": "mean: 261.634234764 usec\nrounds: 3676"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[False-None]",
            "value": 291.494485638,
            "unit": "iter/sec",
            "range": "stddev: 0.0000412361033146",
            "extra": "mean: 3.43059662968 msec\nrounds: 289"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[default-None]",
            "value": 240.594392499,
            "unit": "iter/sec",
            "range": "stddev: 0.0391769705087",
            "extra": "mean: 4.15637284649 msec\nrounds: 417"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[reduce-overhead-None]",
            "value": 453.834642601,
            "unit": "iter/sec",
            "range": "stddev: 0.000238801994929",
            "extra": "mean: 2.20344571818 msec\nrounds: 429"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[False-False-None]",
            "value": 504.12125129,
            "unit": "iter/sec",
            "range": "stddev: 0.0000352743237506",
            "extra": "mean: 1.98364976171 msec\nrounds: 474"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-False-None]",
            "value": 648.298784325,
            "unit": "iter/sec",
            "range": "stddev: 0.0000581789670883",
            "extra": "mean: 1.54249865059 msec\nrounds: 647"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[within-False-None]",
            "value": 191.295698605,
            "unit": "iter/sec",
            "range": "stddev: 0.000110654114627",
            "extra": "mean: 5.22750907255 msec\nrounds: 193"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-default-None]",
            "value": 109.839781201,
            "unit": "iter/sec",
            "range": "stddev: 0.000489888668621",
            "extra": "mean: 9.10416962842 msec\nrounds: 105"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[False-False-None]",
            "value": 138.954370593,
            "unit": "iter/sec",
            "range": "stddev: 0.000170419732902",
            "extra": "mean: 7.19660702811 msec\nrounds: 142"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[True-False-None]",
            "value": 263.307760922,
            "unit": "iter/sec",
            "range": "stddev: 0.0000734596045424",
            "extra": "mean: 3.79783716401 msec\nrounds: 262"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[within-False-None]",
            "value": 107.891129717,
            "unit": "iter/sec",
            "range": "stddev: 0.000895408461871",
            "extra": "mean: 9.26860254988 msec\nrounds: 111"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 10.8451157761,
            "unit": "iter/sec",
            "range": "stddev: 0.00100262249251",
            "extra": "mean: 92.2074066009 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_archive",
            "value": 2.94106365313,
            "unit": "iter/sec",
            "range": "stddev: 0.0832060098033",
            "extra": "mean: 340.013042199 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_zarr",
            "value": 0.44416260945,
            "unit": "iter/sec",
            "range": "stddev: 0.077132987739",
            "extra": "mean: 2.251427695 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 43.1658109633,
            "unit": "iter/sec",
            "range": "stddev: 0.000355007958426",
            "extra": "mean: 23.1664824008 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 6.57580212763,
            "unit": "iter/sec",
            "range": "stddev: 0.0985837446578",
            "extra": "mean: 152.072702401 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 9.94658308901,
            "unit": "iter/sec",
            "range": "stddev: 0.00215615210092",
            "extra": "mean: 100.5370378 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 43.1155046331,
            "unit": "iter/sec",
            "range": "stddev: 0.000326486832151",
            "extra": "mean: 23.1935126008 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-compile]",
            "value": 4798.19404841,
            "unit": "iter/sec",
            "range": "stddev: 0.00000624843368326",
            "extra": "mean: 208.411746151 usec\nrounds: 1761"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-eager]",
            "value": 8165.99758573,
            "unit": "iter/sec",
            "range": "stddev: 0.00000560301272508",
            "extra": "mean: 122.459012447 usec\nrounds: 6589"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-compile]",
            "value": 6558.16456488,
            "unit": "iter/sec",
            "range": "stddev: 0.00000571224747159",
            "extra": "mean: 152.481687537 usec\nrounds: 3046"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-eager]",
            "value": 5721.87687753,
            "unit": "iter/sec",
            "range": "stddev: 0.0000138229577902",
            "extra": "mean: 174.767829054 usec\nrounds: 5341"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-compile]",
            "value": 20990.5973758,
            "unit": "iter/sec",
            "range": "stddev: 0.00000271082313082",
            "extra": "mean: 47.640378313 usec\nrounds: 16407"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-eager]",
            "value": 30289.7330998,
            "unit": "iter/sec",
            "range": "stddev: 0.00000161240218499",
            "extra": "mean: 33.0144870113 usec\nrounds: 25569"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-compile]",
            "value": 50918.390569,
            "unit": "iter/sec",
            "range": "stddev: 0.00000475718179666",
            "extra": "mean: 19.6392696004 usec\nrounds: 38535"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-eager]",
            "value": 15716.5324911,
            "unit": "iter/sec",
            "range": "stddev: 0.0000117670807264",
            "extra": "mean: 63.6272664193 usec\nrounds: 14629"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-compile]",
            "value": 3736.8627471,
            "unit": "iter/sec",
            "range": "stddev: 0.00000795242423076",
            "extra": "mean: 267.604155592 usec\nrounds: 1344"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-eager]",
            "value": 5307.72477831,
            "unit": "iter/sec",
            "range": "stddev: 0.00000578617049526",
            "extra": "mean: 188.404644507 usec\nrounds: 5078"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-compile]",
            "value": 5704.70327867,
            "unit": "iter/sec",
            "range": "stddev: 0.00000751919370492",
            "extra": "mean: 175.293955032 usec\nrounds: 2954"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-eager]",
            "value": 16212.640855,
            "unit": "iter/sec",
            "range": "stddev: 0.00000355748988131",
            "extra": "mean: 61.6802659692 usec\nrounds: 14023"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-compile]",
            "value": 4062.04557771,
            "unit": "iter/sec",
            "range": "stddev: 0.00000786473324902",
            "extra": "mean: 246.18138346 usec\nrounds: 1229"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-eager]",
            "value": 1906.72405185,
            "unit": "iter/sec",
            "range": "stddev: 0.0000162459383118",
            "extra": "mean: 524.459739745 usec\nrounds: 1691"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-eager]",
            "value": 4259.36494413,
            "unit": "iter/sec",
            "range": "stddev: 0.00000903350471876",
            "extra": "mean: 234.776783186 usec\nrounds: 3833"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-compile]",
            "value": 3663.17385628,
            "unit": "iter/sec",
            "range": "stddev: 0.00000824361237584",
            "extra": "mean: 272.987316254 usec\nrounds: 1173"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-eager]",
            "value": 13155.6115831,
            "unit": "iter/sec",
            "range": "stddev: 0.00000409972673051",
            "extra": "mean: 76.0131897848 usec\nrounds: 11366"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-compile]",
            "value": 5777.48322602,
            "unit": "iter/sec",
            "range": "stddev: 0.0000071798964751",
            "extra": "mean: 173.085747008 usec\nrounds: 3170"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-eager]",
            "value": 2186.35896348,
            "unit": "iter/sec",
            "range": "stddev: 0.0000229330206485",
            "extra": "mean: 457.381434934 usec\nrounds: 2150"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-compile]",
            "value": 4095.73637036,
            "unit": "iter/sec",
            "range": "stddev: 0.00000989761169221",
            "extra": "mean: 244.156339561 usec\nrounds: 1401"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-compile]",
            "value": 33620.487249,
            "unit": "iter/sec",
            "range": "stddev: 0.00000221573206218",
            "extra": "mean: 29.7437688096 usec\nrounds: 21552"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-eager]",
            "value": 55350.8204747,
            "unit": "iter/sec",
            "range": "stddev: 0.00000115250555192",
            "extra": "mean: 18.0665795272 usec\nrounds: 46491"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-compile]",
            "value": 34511.5245985,
            "unit": "iter/sec",
            "range": "stddev: 0.00000252123833079",
            "extra": "mean: 28.9758279773 usec\nrounds: 29044"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-eager]",
            "value": 19810.8759743,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113280903474",
            "extra": "mean: 50.4773237336 usec\nrounds: 17337"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-compile]",
            "value": 4160.13749106,
            "unit": "iter/sec",
            "range": "stddev: 0.000233652413576",
            "extra": "mean: 240.376670759 usec\nrounds: 88"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-eager]",
            "value": 304.856295251,
            "unit": "iter/sec",
            "range": "stddev: 0.0000390982163738",
            "extra": "mean: 3.28023404987 msec\nrounds: 302"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-compile]",
            "value": 5084.0566832,
            "unit": "iter/sec",
            "range": "stddev: 0.000209994386451",
            "extra": "mean: 196.693322343 usec\nrounds: 90"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-eager]",
            "value": 349.494739832,
            "unit": "iter/sec",
            "range": "stddev: 0.0000441854902975",
            "extra": "mean: 2.86127339279 msec\nrounds: 349"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-compile]",
            "value": 6285.46900575,
            "unit": "iter/sec",
            "range": "stddev: 0.00000544358777113",
            "extra": "mean: 159.097117349 usec\nrounds: 3905"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-eager]",
            "value": 17799.2375181,
            "unit": "iter/sec",
            "range": "stddev: 0.0000125240113649",
            "extra": "mean: 56.1821819043 usec\nrounds: 15399"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-compile]",
            "value": 7665.87898413,
            "unit": "iter/sec",
            "range": "stddev: 0.00000592121272645",
            "extra": "mean: 130.448185012 usec\nrounds: 4384"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-eager]",
            "value": 21325.7569363,
            "unit": "iter/sec",
            "range": "stddev: 0.000011651526301",
            "extra": "mean: 46.8916532711 usec\nrounds: 18015"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-compile]",
            "value": 7521.26249449,
            "unit": "iter/sec",
            "range": "stddev: 0.00000586793987758",
            "extra": "mean: 132.956402031 usec\nrounds: 6968"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-eager]",
            "value": 20218.4834488,
            "unit": "iter/sec",
            "range": "stddev: 0.0000116460054063",
            "extra": "mean: 49.4596937764 usec\nrounds: 17668"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-compile]",
            "value": 11529.069108,
            "unit": "iter/sec",
            "range": "stddev: 0.0000081501200066",
            "extra": "mean: 86.7372717285 usec\nrounds: 9774"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-eager]",
            "value": 45477.4842806,
            "unit": "iter/sec",
            "range": "stddev: 0.0000119435255644",
            "extra": "mean: 21.9889032082 usec\nrounds: 33806"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-compile]",
            "value": 15461.5740278,
            "unit": "iter/sec",
            "range": "stddev: 0.00000635323760156",
            "extra": "mean: 64.67646814 usec\nrounds: 12869"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-eager]",
            "value": 39648.0060292,
            "unit": "iter/sec",
            "range": "stddev: 0.0000127547459122",
            "extra": "mean: 25.2219493526 usec\nrounds: 35816"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-compile]",
            "value": 15432.3068518,
            "unit": "iter/sec",
            "range": "stddev: 0.00000574560803198",
            "extra": "mean: 64.7991262488 usec\nrounds: 12663"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-eager]",
            "value": 46708.5009069,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113113009085",
            "extra": "mean: 21.4093790335 usec\nrounds: 36602"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-compile]",
            "value": 11713.1511429,
            "unit": "iter/sec",
            "range": "stddev: 0.00000759155950946",
            "extra": "mean: 85.374122454 usec\nrounds: 10214"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-eager]",
            "value": 46225.9945067,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113127803656",
            "extra": "mean: 21.6328498861 usec\nrounds: 36656"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-compile]",
            "value": 15877.6577885,
            "unit": "iter/sec",
            "range": "stddev: 0.00000549899954809",
            "extra": "mean: 62.9815816236 usec\nrounds: 13725"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-eager]",
            "value": 46893.9206114,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113123688728",
            "extra": "mean: 21.3247258272 usec\nrounds: 36696"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-compile]",
            "value": 15784.5574358,
            "unit": "iter/sec",
            "range": "stddev: 0.00000373369958467",
            "extra": "mean: 63.3530590937 usec\nrounds: 11398"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-eager]",
            "value": 46510.5045988,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113273423063",
            "extra": "mean: 21.5005192617 usec\nrounds: 37271"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-eager]",
            "value": 32873.9572642,
            "unit": "iter/sec",
            "range": "stddev: 0.00000202904189831",
            "extra": "mean: 30.4192157933 usec\nrounds: 23127"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-compile]",
            "value": 6748.37128119,
            "unit": "iter/sec",
            "range": "stddev: 0.0000080690539667",
            "extra": "mean: 148.183903691 usec\nrounds: 4297"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-eager]",
            "value": 2560.65271156,
            "unit": "iter/sec",
            "range": "stddev: 0.0000145345183271",
            "extra": "mean: 390.525429507 usec\nrounds: 2547"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-compile]",
            "value": 6476.88315485,
            "unit": "iter/sec",
            "range": "stddev: 0.00000708649909631",
            "extra": "mean: 154.39525094 usec\nrounds: 3530"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[eager]",
            "value": 5999.23269831,
            "unit": "iter/sec",
            "range": "stddev: 0.00000579825201716",
            "extra": "mean: 166.687983329 usec\nrounds: 5819"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[compile]",
            "value": 5773.33961169,
            "unit": "iter/sec",
            "range": "stddev: 0.00000663661789031",
            "extra": "mean: 173.209973301 usec\nrounds: 3825"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[eager]",
            "value": 5709.31729557,
            "unit": "iter/sec",
            "range": "stddev: 0.00000541200763498",
            "extra": "mean: 175.152290235 usec\nrounds: 5357"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[compile]",
            "value": 5031.12555039,
            "unit": "iter/sec",
            "range": "stddev: 0.00000715900670195",
            "extra": "mean: 198.762680435 usec\nrounds: 3452"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-eager]",
            "value": 108235.23577,
            "unit": "iter/sec",
            "range": "stddev: 6.25209629968e-7",
            "extra": "mean: 9.23913541543 usec\nrounds: 82170"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-compile]",
            "value": 31744.8567668,
            "unit": "iter/sec",
            "range": "stddev: 0.00000518402630466",
            "extra": "mean: 31.5011659161 usec\nrounds: 25368"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-eager]",
            "value": 65014.4042327,
            "unit": "iter/sec",
            "range": "stddev: 0.00000128282559799",
            "extra": "mean: 15.381206854 usec\nrounds: 55341"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-compile]",
            "value": 32657.0671594,
            "unit": "iter/sec",
            "range": "stddev: 0.00000364319537555",
            "extra": "mean: 30.621243332 usec\nrounds: 21381"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-eager]",
            "value": 36818.0458784,
            "unit": "iter/sec",
            "range": "stddev: 0.00000185230478502",
            "extra": "mean: 27.1605941093 usec\nrounds: 33613"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-compile]",
            "value": 22516.5219218,
            "unit": "iter/sec",
            "range": "stddev: 0.00000255077793975",
            "extra": "mean: 44.4118324968 usec\nrounds: 17224"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[eager]",
            "value": 37717.076275,
            "unit": "iter/sec",
            "range": "stddev: 0.00000174835471861",
            "extra": "mean: 26.5131897475 usec\nrounds: 29078"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[compile]",
            "value": 8526.26405338,
            "unit": "iter/sec",
            "range": "stddev: 0.0000075640011729",
            "extra": "mean: 117.284662279 usec\nrounds: 6668"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[eager]",
            "value": 32773.9272324,
            "unit": "iter/sec",
            "range": "stddev: 0.00000326829289786",
            "extra": "mean: 30.512058958 usec\nrounds: 21645"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile]",
            "value": 9103.37126027,
            "unit": "iter/sec",
            "range": "stddev: 0.0000244845458892",
            "extra": "mean: 109.84941418 usec\nrounds: 5521"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile-overhead]",
            "value": 5136.39150523,
            "unit": "iter/sec",
            "range": "stddev: 0.0000574803176785",
            "extra": "mean: 194.689209143 usec\nrounds: 306"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[eager]",
            "value": 3568.52287859,
            "unit": "iter/sec",
            "range": "stddev: 0.0000276425931755",
            "extra": "mean: 280.227991812 usec\nrounds: 3438"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile]",
            "value": 3031.37684698,
            "unit": "iter/sec",
            "range": "stddev: 0.0000391330118451",
            "extra": "mean: 329.883102788 usec\nrounds: 2218"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile-overhead]",
            "value": 264.39162555,
            "unit": "iter/sec",
            "range": "stddev: 0.00167635785038",
            "extra": "mean: 3.78226805755 msec\nrounds: 957"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[False]",
            "value": 6126.49852489,
            "unit": "iter/sec",
            "range": "stddev: 0.00000995049249666",
            "extra": "mean: 163.225371872 usec\nrounds: 5048"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[True]",
            "value": 6061.0004135,
            "unit": "iter/sec",
            "range": "stddev: 0.00000942529946896",
            "extra": "mean: 164.989264441 usec\nrounds: 5021"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[eager]",
            "value": 598.80530515,
            "unit": "iter/sec",
            "range": "stddev: 0.00021193357422",
            "extra": "mean: 1.66999188451 msec\nrounds: 622"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile]",
            "value": 701.42366821,
            "unit": "iter/sec",
            "range": "stddev: 0.0000218775650815",
            "extra": "mean: 1.42567188038 msec\nrounds: 594"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile-overhead]",
            "value": 949.163075224,
            "unit": "iter/sec",
            "range": "stddev: 0.000106630570516",
            "extra": "mean: 1.0535597371 msec\nrounds: 80"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[eager]",
            "value": 10229.3179767,
            "unit": "iter/sec",
            "range": "stddev: 0.00000465178495388",
            "extra": "mean: 97.7582280926 usec\nrounds: 8744"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile]",
            "value": 7970.1881901,
            "unit": "iter/sec",
            "range": "stddev: 0.00000514756652547",
            "extra": "mean: 125.467551851 usec\nrounds: 6560"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile-overhead]",
            "value": 4858.22983516,
            "unit": "iter/sec",
            "range": "stddev: 0.00000857542618328",
            "extra": "mean: 205.836288922 usec\nrounds: 3338"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[eager]",
            "value": 2390.3662267,
            "unit": "iter/sec",
            "range": "stddev: 0.0000343714601918",
            "extra": "mean: 418.345937468 usec\nrounds: 2350"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile]",
            "value": 2951.53696334,
            "unit": "iter/sec",
            "range": "stddev: 0.0000101446422251",
            "extra": "mean: 338.806531113 usec\nrounds: 2790"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile-overhead]",
            "value": 3247.06778723,
            "unit": "iter/sec",
            "range": "stddev: 0.0000192213054996",
            "extra": "mean: 307.970164323 usec\nrounds: 2349"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-eager]",
            "value": 1172.49788381,
            "unit": "iter/sec",
            "range": "stddev: 0.0000389606241344",
            "extra": "mean: 852.880004142 usec\nrounds: 1211"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile]",
            "value": 1394.68376086,
            "unit": "iter/sec",
            "range": "stddev: 0.0000249521152505",
            "extra": "mean: 717.008420162 usec\nrounds: 1359"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile-overhead]",
            "value": 1961.633766,
            "unit": "iter/sec",
            "range": "stddev: 0.0000832717679409",
            "extra": "mean: 509.779153139 usec\nrounds: 1174"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-eager]",
            "value": 779.25655154,
            "unit": "iter/sec",
            "range": "stddev: 0.0000179677302423",
            "extra": "mean: 1.28327442102 msec\nrounds: 772"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile]",
            "value": 1402.72761307,
            "unit": "iter/sec",
            "range": "stddev: 0.0000104670071818",
            "extra": "mean: 712.896781018 usec\nrounds: 1324"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile-overhead]",
            "value": 2021.07693838,
            "unit": "iter/sec",
            "range": "stddev: 0.0000123332165105",
            "extra": "mean: 494.785715977 usec\nrounds: 1257"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-eager]",
            "value": 1161.12409494,
            "unit": "iter/sec",
            "range": "stddev: 0.0000325047089659",
            "extra": "mean: 861.234388603 usec\nrounds: 1194"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile]",
            "value": 1436.97743779,
            "unit": "iter/sec",
            "range": "stddev: 0.0000190606735954",
            "extra": "mean: 695.905150426 usec\nrounds: 1396"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile-overhead]",
            "value": 2218.56448404,
            "unit": "iter/sec",
            "range": "stddev: 0.0000142429581706",
            "extra": "mean: 450.741913159 usec\nrounds: 1115"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-eager]",
            "value": 755.713844795,
            "unit": "iter/sec",
            "range": "stddev: 0.0000275574546896",
            "extra": "mean: 1.32325218982 msec\nrounds: 743"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile]",
            "value": 1232.88364247,
            "unit": "iter/sec",
            "range": "stddev: 0.0000146098591299",
            "extra": "mean: 811.106551791 usec\nrounds: 1167"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile-overhead]",
            "value": 1321.06636579,
            "unit": "iter/sec",
            "range": "stddev: 0.0000933079031536",
            "extra": "mean: 756.964241842 usec\nrounds: 914"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[eager]",
            "value": 345.951425657,
            "unit": "iter/sec",
            "range": "stddev: 0.0000458591777439",
            "extra": "mean: 2.89057921383 msec\nrounds: 337"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile]",
            "value": 1326.41778268,
            "unit": "iter/sec",
            "range": "stddev: 0.0000233204621965",
            "extra": "mean: 753.91027854 usec\nrounds: 1037"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile-overhead]",
            "value": 1938.06634727,
            "unit": "iter/sec",
            "range": "stddev: 0.0000367737146375",
            "extra": "mean: 515.978207561 usec\nrounds: 130"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 991.478269841,
            "unit": "iter/sec",
            "range": "stddev: 0.0000748200384517",
            "extra": "mean: 1.00859497421 msec\nrounds: 972"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1011.94718494,
            "unit": "iter/sec",
            "range": "stddev: 0.0000670721910911",
            "extra": "mean: 988.193865131 usec\nrounds: 926"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1122.68396433,
            "unit": "iter/sec",
            "range": "stddev: 0.000041773553384",
            "extra": "mean: 890.722618095 usec\nrounds: 1136"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1123.05043336,
            "unit": "iter/sec",
            "range": "stddev: 0.0000377010647055",
            "extra": "mean: 890.431961284 usec\nrounds: 1134"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-True]",
            "value": 37.8371813114,
            "unit": "iter/sec",
            "range": "stddev: 0.000142669137438",
            "extra": "mean: 26.4290300001 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-False]",
            "value": 37.8477696084,
            "unit": "iter/sec",
            "range": "stddev: 0.000205300628369",
            "extra": "mean: 26.4216362113 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-True]",
            "value": 37.7435944024,
            "unit": "iter/sec",
            "range": "stddev: 0.000417556694085",
            "extra": "mean: 26.4945619471 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-False]",
            "value": 38.2528854933,
            "unit": "iter/sec",
            "range": "stddev: 0.000196824194732",
            "extra": "mean: 26.1418187701 msec\nrounds: 39"
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
          "id": "5b4785d5114efbfba938591f7aecd6c49c17e0d8",
          "message": "[BugFix] Keyword tensor operands give the dense result on a lazy stack (#3060)",
          "timestamp": "2026-10-19T11:15:30-04:00",
          "tree_id": "e1f4d770cbff972d603299cd2593d6d245292a82",
          "url": "https://github.com/pytorch/tensordict/commit/5b4785d5114efbfba938591f7aecd6c49c17e0d8"
        },
        "date": 1792424296949,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-2]",
            "value": 4534.46319013,
            "unit": "iter/sec",
            "range": "stddev: 0.00000709403006911",
            "extra": "mean: 220.5332711 usec\nrounds: 140"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-8]",
            "value": 3472.14694994,
            "unit": "iter/sec",
            "range": "stddev: 0.0000145993466587",
            "extra": "mean: 288.006243519 usec\nrounds: 3162"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-32]",
            "value": 1965.51079287,
            "unit": "iter/sec",
            "range": "stddev: 0.0000170237191482",
            "extra": "mean: 508.773599019 usec\nrounds: 1786"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-2]",
            "value": 4321.56720456,
            "unit": "iter/sec",
            "range": "stddev: 0.0000105034302527",
            "extra": "mean: 231.397535353 usec\nrounds: 4012"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-8]",
            "value": 3424.16616232,
            "unit": "iter/sec",
            "range": "stddev: 0.00000949103843672",
            "extra": "mean: 292.041902348 usec\nrounds: 3203"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-32]",
            "value": 1778.84146212,
            "unit": "iter/sec",
            "range": "stddev: 0.0000174680315677",
            "extra": "mean: 562.163644875 usec\nrounds: 1642"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-2]",
            "value": 2292.8135938,
            "unit": "iter/sec",
            "range": "stddev: 0.0000281649027809",
            "extra": "mean: 436.145355515 usec\nrounds: 76"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-8]",
            "value": 589.178561193,
            "unit": "iter/sec",
            "range": "stddev: 0.0000447390406408",
            "extra": "mean: 1.69727832251 msec\nrounds: 608"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-32]",
            "value": 155.223182862,
            "unit": "iter/sec",
            "range": "stddev: 0.000165869723185",
            "extra": "mean: 6.44233665076 msec\nrounds: 149"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-2]",
            "value": 2232.5465487,
            "unit": "iter/sec",
            "range": "stddev: 0.0000114730430301",
            "extra": "mean: 447.91899214 usec\nrounds: 1883"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-8]",
            "value": 595.589686607,
            "unit": "iter/sec",
            "range": "stddev: 0.0000331737592569",
            "extra": "mean: 1.67900825432 msec\nrounds: 601"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-32]",
            "value": 152.617017323,
            "unit": "iter/sec",
            "range": "stddev: 0.00015026172715",
            "extra": "mean: 6.55234925659 msec\nrounds: 156"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-2]",
            "value": 3063.86573146,
            "unit": "iter/sec",
            "range": "stddev: 0.0000289580685137",
            "extra": "mean: 326.385059806 usec\nrounds: 2423"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-8]",
            "value": 821.505047257,
            "unit": "iter/sec",
            "range": "stddev: 0.0000816720540384",
            "extra": "mean: 1.21727797454 msec\nrounds: 825"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-32]",
            "value": 210.916706061,
            "unit": "iter/sec",
            "range": "stddev: 0.000274309586448",
            "extra": "mean: 4.74120812276 msec\nrounds: 212"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-2]",
            "value": 3099.87703265,
            "unit": "iter/sec",
            "range": "stddev: 0.0000156675195012",
            "extra": "mean: 322.593441439 usec\nrounds: 2913"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-8]",
            "value": 839.166805825,
            "unit": "iter/sec",
            "range": "stddev: 0.0000423378585036",
            "extra": "mean: 1.19165819365 msec\nrounds: 827"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-32]",
            "value": 213.845503376,
            "unit": "iter/sec",
            "range": "stddev: 0.00015445497707",
            "extra": "mean: 4.67627321694 msec\nrounds: 212"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-2]",
            "value": 3703.93690266,
            "unit": "iter/sec",
            "range": "stddev: 0.00000731536599458",
            "extra": "mean: 269.983000866 usec\nrounds: 3043"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-8]",
            "value": 992.558764429,
            "unit": "iter/sec",
            "range": "stddev: 0.0000380353938484",
            "extra": "mean: 1.00749702268 msec\nrounds: 970"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-32]",
            "value": 253.936908064,
            "unit": "iter/sec",
            "range": "stddev: 0.000119496655843",
            "extra": "mean: 3.93798604395 msec\nrounds: 250"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-2]",
            "value": 3614.88419127,
            "unit": "iter/sec",
            "range": "stddev: 0.0000340549265789",
            "extra": "mean: 276.634035031 usec\nrounds: 3256"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-8]",
            "value": 984.242458088,
            "unit": "iter/sec",
            "range": "stddev: 0.0000626072128285",
            "extra": "mean: 1.01600981728 msec\nrounds: 980"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-32]",
            "value": 235.645775369,
            "unit": "iter/sec",
            "range": "stddev: 0.000549343375505",
            "extra": "mean: 4.24365766131 msec\nrounds: 251"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-128]",
            "value": 179.925688873,
            "unit": "iter/sec",
            "range": "stddev: 0.000522846002514",
            "extra": "mean: 5.55785005613 msec\nrounds: 160"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-256]",
            "value": 91.3852810799,
            "unit": "iter/sec",
            "range": "stddev: 0.00111727469364",
            "extra": "mean: 10.9426812303 msec\nrounds: 87"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-128]",
            "value": 151.951393192,
            "unit": "iter/sec",
            "range": "stddev: 0.000413753582325",
            "extra": "mean: 6.58105186792 msec\nrounds: 144"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-256]",
            "value": 87.6985710236,
            "unit": "iter/sec",
            "range": "stddev: 0.000786301045559",
            "extra": "mean: 11.4026943464 msec\nrounds: 75"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-128]",
            "value": 12.5155471977,
            "unit": "iter/sec",
            "range": "stddev: 0.00260838976319",
            "extra": "mean: 79.9006215395 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-256]",
            "value": 6.29542734278,
            "unit": "iter/sec",
            "range": "stddev: 0.00159779341143",
            "extra": "mean: 158.845451715 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-128]",
            "value": 12.5864378095,
            "unit": "iter/sec",
            "range": "stddev: 0.00191766523467",
            "extra": "mean: 79.4505971538 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-256]",
            "value": 5.53677479084,
            "unit": "iter/sec",
            "range": "stddev: 0.0151562870822",
            "extra": "mean: 180.6105608 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-128]",
            "value": 23.0957019606,
            "unit": "iter/sec",
            "range": "stddev: 0.00102353759048",
            "extra": "mean: 43.2980994345 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-256]",
            "value": 11.6745579786,
            "unit": "iter/sec",
            "range": "stddev: 0.00138421256408",
            "extra": "mean: 85.656347918 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-128]",
            "value": 22.1769020722,
            "unit": "iter/sec",
            "range": "stddev: 0.00427764551143",
            "extra": "mean: 45.0919608495 msec\nrounds: 20"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-256]",
            "value": 11.8221825406,
            "unit": "iter/sec",
            "range": "stddev: 0.00084151933731",
            "extra": "mean: 84.5867500836 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-128]",
            "value": 27.5295381423,
            "unit": "iter/sec",
            "range": "stddev: 0.000627027210901",
            "extra": "mean: 36.3246195715 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-256]",
            "value": 13.8033939536,
            "unit": "iter/sec",
            "range": "stddev: 0.000300275108239",
            "extra": "mean: 72.4459508555 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-128]",
            "value": 27.4236036506,
            "unit": "iter/sec",
            "range": "stddev: 0.000149998661838",
            "extra": "mean: 36.46493775 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-256]",
            "value": 13.7865389637,
            "unit": "iter/sec",
            "range": "stddev: 0.000247273955863",
            "extra": "mean: 72.5345210012 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_mask_many_leaves[cuda]",
            "value": 6202.6989025,
            "unit": "iter/sec",
            "range": "stddev: 0.0000208911016766",
            "extra": "mean: 161.220142348 usec\nrounds: 42"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-None]",
            "value": 6725.66291712,
            "unit": "iter/sec",
            "range": "stddev: 0.00000911017760797",
            "extra": "mean: 148.684228205 usec\nrounds: 1318"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-cpu]",
            "value": 6682.57607708,
            "unit": "iter/sec",
            "range": "stddev: 0.00000567992872217",
            "extra": "mean: 149.642890476 usec\nrounds: 5005"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to",
            "value": 22036.0910905,
            "unit": "iter/sec",
            "range": "stddev: 0.00000301918810386",
            "extra": "mean: 45.380099215 usec\nrounds: 11513"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_nonblocking",
            "value": 22389.51234,
            "unit": "iter/sec",
            "range": "stddev: 0.00000331244374127",
            "extra": "mean: 44.6637686795 usec\nrounds: 11573"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[add]",
            "value": 10363.9431227,
            "unit": "iter/sec",
            "range": "stddev: 0.00000877335871818",
            "extra": "mean: 96.4883720565 usec\nrounds: 188"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[sub]",
            "value": 10431.8527701,
            "unit": "iter/sec",
            "range": "stddev: 0.00000956234659113",
            "extra": "mean: 95.8602486095 usec\nrounds: 7917"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_blocking",
            "value": 34.3615852828,
            "unit": "iter/sec",
            "range": "stddev: 0.000219550789146",
            "extra": "mean: 29.1022661431 msec\nrounds: 35"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_global_sync",
            "value": 94.8659219893,
            "unit": "iter/sec",
            "range": "stddev: 0.0120869732303",
            "extra": "mean: 10.5411930758 msec\nrounds: 106"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_event_sync",
            "value": 96.9063216064,
            "unit": "iter/sec",
            "range": "stddev: 0.000103629321348",
            "extra": "mean: 10.3192442291 msec\nrounds: 96"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_default",
            "value": 94.7993434469,
            "unit": "iter/sec",
            "range": "stddev: 0.000104195832904",
            "extra": "mean: 10.5485962628 msec\nrounds: 95"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_cpu_to_cpu_default",
            "value": 3859.01486878,
            "unit": "iter/sec",
            "range": "stddev: 0.00000473875667528",
            "extra": "mean: 259.133492356 usec\nrounds: 3782"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[False-None]",
            "value": 291.206977567,
            "unit": "iter/sec",
            "range": "stddev: 0.0000372904866996",
            "extra": "mean: 3.43398365092 msec\nrounds: 252"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[default-None]",
            "value": 440.208087639,
            "unit": "iter/sec",
            "range": "stddev: 0.000208134506967",
            "extra": "mean: 2.27165294796 msec\nrounds: 441"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[reduce-overhead-None]",
            "value": 433.701833167,
            "unit": "iter/sec",
            "range": "stddev: 0.000285717175233",
            "extra": "mean: 2.30573154994 msec\nrounds: 422"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[False-False-None]",
            "value": 472.297118356,
            "unit": "iter/sec",
            "range": "stddev: 0.0000959491806463",
            "extra": "mean: 2.11731124568 msec\nrounds: 480"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-False-None]",
            "value": 624.687871834,
            "unit": "iter/sec",
            "range": "stddev: 0.0000660349287718",
            "extra": "mean: 1.60079944735 msec\nrounds: 628"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[within-False-None]",
            "value": 186.481829969,
            "unit": "iter/sec",
            "range": "stddev: 0.000374944139761",
            "extra": "mean: 5.36245273959 msec\nrounds: 192"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-default-None]",
            "value": 111.651461095,
            "unit": "iter/sec",
            "range": "stddev: 0.000209232137478",
            "extra": "mean: 8.95644347319 msec\nrounds: 112"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[False-False-None]",
            "value": 138.746216069,
            "unit": "iter/sec",
            "range": "stddev: 0.0000939420745316",
            "extra": "mean: 7.20740376445 msec\nrounds: 140"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[True-False-None]",
            "value": 253.51551476,
            "unit": "iter/sec",
            "range": "stddev: 0.000201999244607",
            "extra": "mean: 3.94453176148 msec\nrounds: 260"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[within-False-None]",
            "value": 102.389409375,
            "unit": "iter/sec",
            "range": "stddev: 0.00108815505337",
            "extra": "mean: 9.76663510519 msec\nrounds: 95"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 10.6305911687,
            "unit": "iter/sec",
            "range": "stddev: 0.00265845850843",
            "extra": "mean: 94.0681458 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_archive",
            "value": 4.35434900399,
            "unit": "iter/sec",
            "range": "stddev: 0.00387206380292",
            "extra": "mean: 229.6554546 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_zarr",
            "value": 0.448221498845,
            "unit": "iter/sec",
            "range": "stddev: 0.0202816956349",
            "extra": "mean: 2.231039793 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 41.7531949428,
            "unit": "iter/sec",
            "range": "stddev: 0.000272464620458",
            "extra": "mean: 23.9502630007 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.43322456825,
            "unit": "iter/sec",
            "range": "stddev: 0.029853355273",
            "extra": "mean: 106.008289399 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 10.120886807,
            "unit": "iter/sec",
            "range": "stddev: 0.00209894701258",
            "extra": "mean: 98.8055710011 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 40.5934542163,
            "unit": "iter/sec",
            "range": "stddev: 0.00135488926805",
            "extra": "mean: 24.6345136009 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-compile]",
            "value": 4802.93607144,
            "unit": "iter/sec",
            "range": "stddev: 0.00000737002942819",
            "extra": "mean: 208.205977578 usec\nrounds: 1791"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-eager]",
            "value": 8071.73370403,
            "unit": "iter/sec",
            "range": "stddev: 0.000011351904239",
            "extra": "mean: 123.889121799 usec\nrounds: 6369"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-compile]",
            "value": 6451.89121853,
            "unit": "iter/sec",
            "range": "stddev: 0.00000748715927948",
            "extra": "mean: 154.993313763 usec\nrounds: 3002"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-eager]",
            "value": 5793.87327305,
            "unit": "iter/sec",
            "range": "stddev: 0.0000144274326034",
            "extra": "mean: 172.59611194 usec\nrounds: 5397"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-compile]",
            "value": 20246.2703478,
            "unit": "iter/sec",
            "range": "stddev: 0.00000267147693023",
            "extra": "mean: 49.3918130512 usec\nrounds: 15714"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-eager]",
            "value": 30658.0268092,
            "unit": "iter/sec",
            "range": "stddev: 0.00000126230401269",
            "extra": "mean: 32.6178852352 usec\nrounds: 25913"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-compile]",
            "value": 57890.4756027,
            "unit": "iter/sec",
            "range": "stddev: 0.00000360851253533",
            "extra": "mean: 17.2739986948 usec\nrounds: 36311"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-eager]",
            "value": 15452.3632379,
            "unit": "iter/sec",
            "range": "stddev: 0.0000117566699684",
            "extra": "mean: 64.7150202595 usec\nrounds: 14443"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-compile]",
            "value": 3699.05354863,
            "unit": "iter/sec",
            "range": "stddev: 0.0000058452482378",
            "extra": "mean: 270.339422464 usec\nrounds: 1328"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-eager]",
            "value": 4640.5673469,
            "unit": "iter/sec",
            "range": "stddev: 0.000035646437961",
            "extra": "mean: 215.490892653 usec\nrounds: 5004"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-compile]",
            "value": 5725.54694277,
            "unit": "iter/sec",
            "range": "stddev: 0.00000481286525448",
            "extra": "mean: 174.655803192 usec\nrounds: 3242"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-eager]",
            "value": 15851.9645027,
            "unit": "iter/sec",
            "range": "stddev: 0.00000586837676442",
            "extra": "mean: 63.0836638467 usec\nrounds: 13541"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-compile]",
            "value": 4007.50121685,
            "unit": "iter/sec",
            "range": "stddev: 0.0000199646981382",
            "extra": "mean: 249.532051494 usec\nrounds: 1343"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-eager]",
            "value": 1833.53598707,
            "unit": "iter/sec",
            "range": "stddev: 0.000022112816992",
            "extra": "mean: 545.394258446 usec\nrounds: 1919"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-eager]",
            "value": 4247.80658518,
            "unit": "iter/sec",
            "range": "stddev: 0.00000986850954835",
            "extra": "mean: 235.415615082 usec\nrounds: 3123"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-compile]",
            "value": 3610.71800747,
            "unit": "iter/sec",
            "range": "stddev: 0.0000208091250057",
            "extra": "mean: 276.953225905 usec\nrounds: 1274"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-eager]",
            "value": 12697.032255,
            "unit": "iter/sec",
            "range": "stddev: 0.00000738018950678",
            "extra": "mean: 78.7585618368 usec\nrounds: 11222"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-compile]",
            "value": 5790.10276032,
            "unit": "iter/sec",
            "range": "stddev: 0.00000646934986322",
            "extra": "mean: 172.708506463 usec\nrounds: 3201"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-eager]",
            "value": 2251.49956081,
            "unit": "iter/sec",
            "range": "stddev: 0.0000191125260118",
            "extra": "mean: 444.148432185 usec\nrounds: 2219"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-compile]",
            "value": 4109.30954872,
            "unit": "iter/sec",
            "range": "stddev: 0.00000568944520627",
            "extra": "mean: 243.349883513 usec\nrounds: 1332"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-compile]",
            "value": 32214.9751444,
            "unit": "iter/sec",
            "range": "stddev: 0.00000268014922875",
            "extra": "mean: 31.041464273 usec\nrounds: 21983"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-eager]",
            "value": 53740.5204792,
            "unit": "iter/sec",
            "range": "stddev: 0.00000111774935147",
            "extra": "mean: 18.607932917 usec\nrounds: 45912"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-compile]",
            "value": 46035.569488,
            "unit": "iter/sec",
            "range": "stddev: 0.00000137538286895",
            "extra": "mean: 21.722333646 usec\nrounds: 28969"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-eager]",
            "value": 19328.166372,
            "unit": "iter/sec",
            "range": "stddev: 0.0000126010401784",
            "extra": "mean: 51.7379652449 usec\nrounds: 17106"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-compile]",
            "value": 3965.70677209,
            "unit": "iter/sec",
            "range": "stddev: 0.000222784681844",
            "extra": "mean: 252.161861042 usec\nrounds: 94"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-eager]",
            "value": 311.926004513,
            "unit": "iter/sec",
            "range": "stddev: 0.0000493939841261",
            "extra": "mean: 3.20588852975 msec\nrounds: 306"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-compile]",
            "value": 5034.74167019,
            "unit": "iter/sec",
            "range": "stddev: 0.000211260233455",
            "extra": "mean: 198.619922432 usec\nrounds: 90"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-eager]",
            "value": 351.743880492,
            "unit": "iter/sec",
            "range": "stddev: 0.0000401835715949",
            "extra": "mean: 2.84297767626 msec\nrounds: 346"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-compile]",
            "value": 6246.81626149,
            "unit": "iter/sec",
            "range": "stddev: 0.00000663811395763",
            "extra": "mean: 160.081545245 usec\nrounds: 3907"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-eager]",
            "value": 18367.6367188,
            "unit": "iter/sec",
            "range": "stddev: 0.0000121742802278",
            "extra": "mean: 54.4435854927 usec\nrounds: 15596"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-compile]",
            "value": 7533.93078225,
            "unit": "iter/sec",
            "range": "stddev: 0.00000682879446513",
            "extra": "mean: 132.732836139 usec\nrounds: 4376"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-eager]",
            "value": 21986.3032156,
            "unit": "iter/sec",
            "range": "stddev: 0.0000115836481808",
            "extra": "mean: 45.4828622254 usec\nrounds: 17665"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-compile]",
            "value": 7568.79552404,
            "unit": "iter/sec",
            "range": "stddev: 0.00000564927774701",
            "extra": "mean: 132.121418372 usec\nrounds: 6859"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-eager]",
            "value": 21990.4305162,
            "unit": "iter/sec",
            "range": "stddev: 0.0000116869602154",
            "extra": "mean: 45.4743257193 usec\nrounds: 18512"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-compile]",
            "value": 11659.2905566,
            "unit": "iter/sec",
            "range": "stddev: 0.00000488573156514",
            "extra": "mean: 85.7685118273 usec\nrounds: 9613"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-eager]",
            "value": 46114.0677736,
            "unit": "iter/sec",
            "range": "stddev: 0.000011058145009",
            "extra": "mean: 21.6853565144 usec\nrounds: 35001"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-compile]",
            "value": 15574.9943825,
            "unit": "iter/sec",
            "range": "stddev: 0.00000498281556475",
            "extra": "mean: 64.2054806211 usec\nrounds: 9243"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-eager]",
            "value": 47574.7520566,
            "unit": "iter/sec",
            "range": "stddev: 0.0000109087666146",
            "extra": "mean: 21.0195525309 usec\nrounds: 36285"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-compile]",
            "value": 15485.8912845,
            "unit": "iter/sec",
            "range": "stddev: 0.00000384693172346",
            "extra": "mean: 64.5749076775 usec\nrounds: 10207"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-eager]",
            "value": 47387.9082217,
            "unit": "iter/sec",
            "range": "stddev: 0.000010831473081",
            "extra": "mean: 21.1024296604 usec\nrounds: 37763"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-compile]",
            "value": 11492.9404448,
            "unit": "iter/sec",
            "range": "stddev: 0.00000558455916506",
            "extra": "mean: 87.0099349079 usec\nrounds: 9967"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-eager]",
            "value": 47233.5848626,
            "unit": "iter/sec",
            "range": "stddev: 0.0000108712941778",
            "extra": "mean: 21.1713763185 usec\nrounds: 36656"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-compile]",
            "value": 15497.1302455,
            "unit": "iter/sec",
            "range": "stddev: 0.00000390861407384",
            "extra": "mean: 64.5280761119 usec\nrounds: 13332"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-eager]",
            "value": 46944.9775451,
            "unit": "iter/sec",
            "range": "stddev: 0.0000110310575233",
            "extra": "mean: 21.3015332479 usec\nrounds: 40323"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-compile]",
            "value": 15205.2137159,
            "unit": "iter/sec",
            "range": "stddev: 0.00000397979535609",
            "extra": "mean: 65.7669151307 usec\nrounds: 12852"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-eager]",
            "value": 47421.3430215,
            "unit": "iter/sec",
            "range": "stddev: 0.0000110859821187",
            "extra": "mean: 21.0875512224 usec\nrounds: 37793"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-eager]",
            "value": 33367.1220901,
            "unit": "iter/sec",
            "range": "stddev: 0.00000292307602297",
            "extra": "mean: 29.9696209131 usec\nrounds: 23100"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-compile]",
            "value": 6483.58824891,
            "unit": "iter/sec",
            "range": "stddev: 0.0000135861695516",
            "extra": "mean: 154.235580917 usec\nrounds: 4286"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-eager]",
            "value": 2577.93510487,
            "unit": "iter/sec",
            "range": "stddev: 0.0000215779783877",
            "extra": "mean: 387.907359696 usec\nrounds: 2502"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-compile]",
            "value": 5837.41186276,
            "unit": "iter/sec",
            "range": "stddev: 0.0000387421094439",
            "extra": "mean: 171.308796349 usec\nrounds: 2774"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[eager]",
            "value": 5325.40962514,
            "unit": "iter/sec",
            "range": "stddev: 0.0000312375106375",
            "extra": "mean: 187.778982349 usec\nrounds: 3276"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[compile]",
            "value": 5530.46308206,
            "unit": "iter/sec",
            "range": "stddev: 0.00000745684512573",
            "extra": "mean: 180.816684817 usec\nrounds: 3475"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[eager]",
            "value": 5830.11778236,
            "unit": "iter/sec",
            "range": "stddev: 0.00000677296228915",
            "extra": "mean: 171.523121372 usec\nrounds: 5590"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[compile]",
            "value": 4966.00483968,
            "unit": "iter/sec",
            "range": "stddev: 0.000019731186496",
            "extra": "mean: 201.36911507 usec\nrounds: 3409"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-eager]",
            "value": 108581.190563,
            "unit": "iter/sec",
            "range": "stddev: 0.00000141279587015",
            "extra": "mean: 9.20969824344 usec\nrounds: 84176"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-compile]",
            "value": 40507.6865856,
            "unit": "iter/sec",
            "range": "stddev: 0.00000305117565672",
            "extra": "mean: 24.6866726859 usec\nrounds: 24655"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-eager]",
            "value": 65050.1299033,
            "unit": "iter/sec",
            "range": "stddev: 0.00000204920549071",
            "extra": "mean: 15.3727594624 usec\nrounds: 55097"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-compile]",
            "value": 32962.3458841,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031138016884",
            "extra": "mean: 30.3376465837 usec\nrounds: 18660"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-eager]",
            "value": 36337.0693524,
            "unit": "iter/sec",
            "range": "stddev: 0.00000306052167563",
            "extra": "mean: 27.5201059915 usec\nrounds: 32691"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-compile]",
            "value": 23064.4992831,
            "unit": "iter/sec",
            "range": "stddev: 0.00000371420649025",
            "extra": "mean: 43.3566750237 usec\nrounds: 16958"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[eager]",
            "value": 37757.2408172,
            "unit": "iter/sec",
            "range": "stddev: 0.00000416372276746",
            "extra": "mean: 26.4849861472 usec\nrounds: 29113"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[compile]",
            "value": 8233.11799333,
            "unit": "iter/sec",
            "range": "stddev: 0.00000702158205072",
            "extra": "mean: 121.460666641 usec\nrounds: 7074"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[eager]",
            "value": 30216.7667119,
            "unit": "iter/sec",
            "range": "stddev: 0.00000406106274212",
            "extra": "mean: 33.094209236 usec\nrounds: 21778"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile]",
            "value": 9215.16274059,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113039439617",
            "extra": "mean: 108.516803029 usec\nrounds: 5553"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile-overhead]",
            "value": 5233.91143128,
            "unit": "iter/sec",
            "range": "stddev: 0.0000115007789125",
            "extra": "mean: 191.061696998 usec\nrounds: 307"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[eager]",
            "value": 3602.55705121,
            "unit": "iter/sec",
            "range": "stddev: 0.0000127683247567",
            "extra": "mean: 277.580614488 usec\nrounds: 3453"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile]",
            "value": 2999.99760528,
            "unit": "iter/sec",
            "range": "stddev: 0.0000387017082457",
            "extra": "mean: 333.333599413 usec\nrounds: 1705"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile-overhead]",
            "value": 250.835029309,
            "unit": "iter/sec",
            "range": "stddev: 0.0017917704743",
            "extra": "mean: 3.98668400803 msec\nrounds: 1002"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[False]",
            "value": 6251.96979739,
            "unit": "iter/sec",
            "range": "stddev: 0.00000722904512808",
            "extra": "mean: 159.949589075 usec\nrounds: 5225"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[True]",
            "value": 5741.25100201,
            "unit": "iter/sec",
            "range": "stddev: 0.0000155177498774",
            "extra": "mean: 174.178066705 usec\nrounds: 5356"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[eager]",
            "value": 618.926317736,
            "unit": "iter/sec",
            "range": "stddev: 0.0000698034624932",
            "extra": "mean: 1.61570120925 msec\nrounds: 593"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile]",
            "value": 749.806067034,
            "unit": "iter/sec",
            "range": "stddev: 0.0000556911609071",
            "extra": "mean: 1.33367819222 msec\nrounds: 661"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile-overhead]",
            "value": 1038.99635452,
            "unit": "iter/sec",
            "range": "stddev: 0.0000975608534828",
            "extra": "mean: 962.46728456 usec\nrounds: 88"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[eager]",
            "value": 10334.2439882,
            "unit": "iter/sec",
            "range": "stddev: 0.00000387808863009",
            "extra": "mean: 96.7656657941 usec\nrounds: 8664"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile]",
            "value": 7860.1402247,
            "unit": "iter/sec",
            "range": "stddev: 0.00000737566472825",
            "extra": "mean: 127.224193388 usec\nrounds: 6675"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile-overhead]",
            "value": 4706.63888137,
            "unit": "iter/sec",
            "range": "stddev: 0.00000819970895962",
            "extra": "mean: 212.465843504 usec\nrounds: 3226"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[eager]",
            "value": 2370.64951184,
            "unit": "iter/sec",
            "range": "stddev: 0.0000155297241908",
            "extra": "mean: 421.825324666 usec\nrounds: 2310"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile]",
            "value": 2831.31987561,
            "unit": "iter/sec",
            "range": "stddev: 0.000019841059093",
            "extra": "mean: 353.192166174 usec\nrounds: 2581"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile-overhead]",
            "value": 3216.41543733,
            "unit": "iter/sec",
            "range": "stddev: 0.0000139993929535",
            "extra": "mean: 310.905111446 usec\nrounds: 2332"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-eager]",
            "value": 1192.08112341,
            "unit": "iter/sec",
            "range": "stddev: 0.000024635849644",
            "extra": "mean: 838.869083957 usec\nrounds: 1049"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile]",
            "value": 1275.12937131,
            "unit": "iter/sec",
            "range": "stddev: 0.000051194983861",
            "extra": "mean: 784.234151057 usec\nrounds: 1191"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile-overhead]",
            "value": 2096.20012749,
            "unit": "iter/sec",
            "range": "stddev: 0.0000137662002052",
            "extra": "mean: 477.053687234 usec\nrounds: 1240"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-eager]",
            "value": 759.397903112,
            "unit": "iter/sec",
            "range": "stddev: 0.0000298697941254",
            "extra": "mean: 1.31683271168 msec\nrounds: 742"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile]",
            "value": 1338.06829062,
            "unit": "iter/sec",
            "range": "stddev: 0.0000206790305895",
            "extra": "mean: 747.346011416 usec\nrounds: 1302"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile-overhead]",
            "value": 1987.40453471,
            "unit": "iter/sec",
            "range": "stddev: 0.000012275959936",
            "extra": "mean: 503.168822722 usec\nrounds: 1230"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-eager]",
            "value": 1138.95065883,
            "unit": "iter/sec",
            "range": "stddev: 0.0000205354316407",
            "extra": "mean: 878.001160319 usec\nrounds: 1186"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile]",
            "value": 1352.91008305,
            "unit": "iter/sec",
            "range": "stddev: 0.000023610554998",
            "extra": "mean: 739.147421936 usec\nrounds: 1306"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile-overhead]",
            "value": 2158.78034023,
            "unit": "iter/sec",
            "range": "stddev: 0.0000218464430269",
            "extra": "mean: 463.224526074 usec\nrounds: 1091"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-eager]",
            "value": 737.195388126,
            "unit": "iter/sec",
            "range": "stddev: 0.0000163287483495",
            "extra": "mean: 1.35649247962 msec\nrounds: 732"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile]",
            "value": 1193.59821746,
            "unit": "iter/sec",
            "range": "stddev: 0.000039435518897",
            "extra": "mean: 837.802859765 usec\nrounds: 1176"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile-overhead]",
            "value": 1320.799437,
            "unit": "iter/sec",
            "range": "stddev: 0.0000884232605608",
            "extra": "mean: 757.117221573 usec\nrounds: 948"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[eager]",
            "value": 332.288648605,
            "unit": "iter/sec",
            "range": "stddev: 0.000166963279846",
            "extra": "mean: 3.00943172209 msec\nrounds: 331"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile]",
            "value": 1283.59603902,
            "unit": "iter/sec",
            "range": "stddev: 0.0000281017292696",
            "extra": "mean: 779.061300906 usec\nrounds: 1018"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile-overhead]",
            "value": 1930.66392942,
            "unit": "iter/sec",
            "range": "stddev: 0.000037237783032",
            "extra": "mean: 517.956535448 usec\nrounds: 125"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1001.579357,
            "unit": "iter/sec",
            "range": "stddev: 0.0000439419207598",
            "extra": "mean: 998.423133434 usec\nrounds: 929"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1022.53470298,
            "unit": "iter/sec",
            "range": "stddev: 0.0000385321548226",
            "extra": "mean: 977.96191864 usec\nrounds: 1020"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1123.68487068,
            "unit": "iter/sec",
            "range": "stddev: 0.0000359075630772",
            "extra": "mean: 889.929219566 usec\nrounds: 1116"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1125.8695926,
            "unit": "iter/sec",
            "range": "stddev: 0.000036176449264",
            "extra": "mean: 888.202334066 usec\nrounds: 1119"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-True]",
            "value": 37.134063103,
            "unit": "iter/sec",
            "range": "stddev: 0.000117946303485",
            "extra": "mean: 26.9294528106 msec\nrounds: 37"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-False]",
            "value": 37.2676080218,
            "unit": "iter/sec",
            "range": "stddev: 0.000133035655417",
            "extra": "mean: 26.832953685 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-True]",
            "value": 37.4719720627,
            "unit": "iter/sec",
            "range": "stddev: 0.0000725109667725",
            "extra": "mean: 26.6866125521 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-False]",
            "value": 37.4474577423,
            "unit": "iter/sec",
            "range": "stddev: 0.000114692916845",
            "extra": "mean: 26.7040824742 msec\nrounds: 38"
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
          "id": "8b8945b9301d4ba92c2dfbc0c9759311541d165e",
          "message": "[Doc] Stop the stable 0.15 docs from saying they are an unstable development version (#3062)",
          "timestamp": "2026-10-19T17:56:14-04:00",
          "tree_id": "d74c4be36ec2186a15412ae0a9daae3497193a4d",
          "url": "https://github.com/pytorch/tensordict/commit/8b8945b9301d4ba92c2dfbc0c9759311541d165e"
        },
        "date": 1792447665642,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-2]",
            "value": 4640.12409612,
            "unit": "iter/sec",
            "range": "stddev: 0.00000634226857237",
            "extra": "mean: 215.511477556 usec\nrounds: 134"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-8]",
            "value": 3688.75206963,
            "unit": "iter/sec",
            "range": "stddev: 0.00000683343902859",
            "extra": "mean: 271.094392121 usec\nrounds: 3262"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-32]",
            "value": 2072.69922486,
            "unit": "iter/sec",
            "range": "stddev: 0.00000856691111257",
            "extra": "mean: 482.462668972 usec\nrounds: 1840"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-2]",
            "value": 4640.56256933,
            "unit": "iter/sec",
            "range": "stddev: 0.0000061257252098",
            "extra": "mean: 215.491114506 usec\nrounds: 3720"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-8]",
            "value": 3669.08898749,
            "unit": "iter/sec",
            "range": "stddev: 0.00000878436966623",
            "extra": "mean: 272.547219054 usec\nrounds: 3191"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-32]",
            "value": 1797.80603964,
            "unit": "iter/sec",
            "range": "stddev: 0.000041742882462",
            "extra": "mean: 556.233530176 usec\nrounds: 1677"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-2]",
            "value": 2324.78611832,
            "unit": "iter/sec",
            "range": "stddev: 0.0000150224299047",
            "extra": "mean: 430.147097025 usec\nrounds: 72"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-8]",
            "value": 624.181270666,
            "unit": "iter/sec",
            "range": "stddev: 0.000024048455224",
            "extra": "mean: 1.60209869632 msec\nrounds: 609"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-32]",
            "value": 158.902271332,
            "unit": "iter/sec",
            "range": "stddev: 0.000068649841155",
            "extra": "mean: 6.2931762499 msec\nrounds: 156"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-2]",
            "value": 2321.54036105,
            "unit": "iter/sec",
            "range": "stddev: 0.0000134127191344",
            "extra": "mean: 430.748487848 usec\nrounds: 2062"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-8]",
            "value": 616.238743453,
            "unit": "iter/sec",
            "range": "stddev: 0.0000352502372136",
            "extra": "mean: 1.62274769418 msec\nrounds: 582"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-32]",
            "value": 157.76206641,
            "unit": "iter/sec",
            "range": "stddev: 0.000102168149264",
            "extra": "mean: 6.33865936696 msec\nrounds: 158"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-2]",
            "value": 3170.60305888,
            "unit": "iter/sec",
            "range": "stddev: 0.0000108878677542",
            "extra": "mean: 315.397412236 usec\nrounds: 2399"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-8]",
            "value": 857.767093066,
            "unit": "iter/sec",
            "range": "stddev: 0.0000179907282369",
            "extra": "mean: 1.16581763055 msec\nrounds: 820"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-32]",
            "value": 218.327648093,
            "unit": "iter/sec",
            "range": "stddev: 0.0000441170471911",
            "extra": "mean: 4.58027193869 msec\nrounds: 212"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-2]",
            "value": 3126.89766479,
            "unit": "iter/sec",
            "range": "stddev: 0.0000186616401798",
            "extra": "mean: 319.805797055 usec\nrounds: 2932"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-8]",
            "value": 838.800612465,
            "unit": "iter/sec",
            "range": "stddev: 0.0000532213974461",
            "extra": "mean: 1.19217843328 msec\nrounds: 824"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-32]",
            "value": 216.487133056,
            "unit": "iter/sec",
            "range": "stddev: 0.0000942535423382",
            "extra": "mean: 4.61921217157 msec\nrounds: 210"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-2]",
            "value": 3662.00124777,
            "unit": "iter/sec",
            "range": "stddev: 0.000011858231806",
            "extra": "mean: 273.074729455 usec\nrounds: 3057"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-8]",
            "value": 1001.15318595,
            "unit": "iter/sec",
            "range": "stddev: 0.0000344111790743",
            "extra": "mean: 998.848142358 usec\nrounds: 976"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-32]",
            "value": 256.681499248,
            "unit": "iter/sec",
            "range": "stddev: 0.0000897199397651",
            "extra": "mean: 3.89587875609 msec\nrounds: 254"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-2]",
            "value": 3732.29863924,
            "unit": "iter/sec",
            "range": "stddev: 0.0000100737794377",
            "extra": "mean: 267.931400099 usec\nrounds: 3404"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-8]",
            "value": 994.244333348,
            "unit": "iter/sec",
            "range": "stddev: 0.0000303231612754",
            "extra": "mean: 1.00578898613 msec\nrounds: 932"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-32]",
            "value": 250.721157958,
            "unit": "iter/sec",
            "range": "stddev: 0.000114678795381",
            "extra": "mean: 3.98849466134 msec\nrounds: 251"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-128]",
            "value": 191.223096579,
            "unit": "iter/sec",
            "range": "stddev: 0.0000309467019738",
            "extra": "mean: 5.22949381059 msec\nrounds: 185"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-256]",
            "value": 98.2077935249,
            "unit": "iter/sec",
            "range": "stddev: 0.000367448732204",
            "extra": "mean: 10.1824912678 msec\nrounds: 97"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-128]",
            "value": 181.845663585,
            "unit": "iter/sec",
            "range": "stddev: 0.0000750390039547",
            "extra": "mean: 5.49916880219 msec\nrounds: 177"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-256]",
            "value": 94.3427199294,
            "unit": "iter/sec",
            "range": "stddev: 0.000198407495942",
            "extra": "mean: 10.5996520002 msec\nrounds: 87"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-128]",
            "value": 13.4904983457,
            "unit": "iter/sec",
            "range": "stddev: 0.000784950661794",
            "extra": "mean: 74.1262460716 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-256]",
            "value": 6.71629427976,
            "unit": "iter/sec",
            "range": "stddev: 0.000821090691269",
            "extra": "mean: 148.891629572 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-128]",
            "value": 13.2941743046,
            "unit": "iter/sec",
            "range": "stddev: 0.0013085254425",
            "extra": "mean: 75.2209183577 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-256]",
            "value": 6.57013839361,
            "unit": "iter/sec",
            "range": "stddev: 0.00112618015781",
            "extra": "mean: 152.203795429 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-128]",
            "value": 23.9702643116,
            "unit": "iter/sec",
            "range": "stddev: 0.000169351039566",
            "extra": "mean: 41.718355167 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-256]",
            "value": 11.9647606809,
            "unit": "iter/sec",
            "range": "stddev: 0.000536468671536",
            "extra": "mean: 83.5787715835 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-128]",
            "value": 23.7168729043,
            "unit": "iter/sec",
            "range": "stddev: 0.000319811677615",
            "extra": "mean: 42.1640746667 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-256]",
            "value": 11.9183862638,
            "unit": "iter/sec",
            "range": "stddev: 0.000394612592081",
            "extra": "mean: 83.9039764166 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-128]",
            "value": 28.5766233794,
            "unit": "iter/sec",
            "range": "stddev: 0.000281534426926",
            "extra": "mean: 34.993637517 msec\nrounds: 29"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-256]",
            "value": 14.3460671518,
            "unit": "iter/sec",
            "range": "stddev: 0.000923875777724",
            "extra": "mean: 69.7055150667 msec\nrounds: 15"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-128]",
            "value": 28.6445307403,
            "unit": "iter/sec",
            "range": "stddev: 0.000234011536276",
            "extra": "mean: 34.9106783793 msec\nrounds: 29"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-256]",
            "value": 14.2728456162,
            "unit": "iter/sec",
            "range": "stddev: 0.000605324289159",
            "extra": "mean: 70.0631133334 msec\nrounds: 15"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_mask_many_leaves[cuda]",
            "value": 6486.16533955,
            "unit": "iter/sec",
            "range": "stddev: 0.0000175635922281",
            "extra": "mean: 154.17429986 usec\nrounds: 40"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-None]",
            "value": 6755.02608692,
            "unit": "iter/sec",
            "range": "stddev: 0.00000943674687263",
            "extra": "mean: 148.037918305 usec\nrounds: 1395"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-cpu]",
            "value": 6677.30335075,
            "unit": "iter/sec",
            "range": "stddev: 0.00000659855437239",
            "extra": "mean: 149.761055844 usec\nrounds: 4675"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to",
            "value": 24545.9042042,
            "unit": "iter/sec",
            "range": "stddev: 0.00000294876288534",
            "extra": "mean: 40.7399944072 usec\nrounds: 11676"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_nonblocking",
            "value": 24764.5398264,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030542545908",
            "extra": "mean: 40.38031827 usec\nrounds: 12018"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[add]",
            "value": 10834.0811445,
            "unit": "iter/sec",
            "range": "stddev: 0.00000736503373193",
            "extra": "mean: 92.3013208656 usec\nrounds: 187"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[sub]",
            "value": 10980.0311787,
            "unit": "iter/sec",
            "range": "stddev: 0.00000571777738294",
            "extra": "mean: 91.0744226248 usec\nrounds: 8135"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_blocking",
            "value": 55.0467653103,
            "unit": "iter/sec",
            "range": "stddev: 0.000261811320207",
            "extra": "mean: 18.1663717089 msec\nrounds: 55"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_global_sync",
            "value": 94.77223901,
            "unit": "iter/sec",
            "range": "stddev: 0.0121891300158",
            "extra": "mean: 10.5516131142 msec\nrounds: 105"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_event_sync",
            "value": 97.1762869798,
            "unit": "iter/sec",
            "range": "stddev: 0.000262961714289",
            "extra": "mean: 10.2905763441 msec\nrounds: 96"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_default",
            "value": 94.8947988707,
            "unit": "iter/sec",
            "range": "stddev: 0.000243886802652",
            "extra": "mean: 10.5379853469 msec\nrounds: 98"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_cpu_to_cpu_default",
            "value": 3879.50770362,
            "unit": "iter/sec",
            "range": "stddev: 0.00000777678660751",
            "extra": "mean: 257.764664075 usec\nrounds: 3858"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[False-None]",
            "value": 292.899942153,
            "unit": "iter/sec",
            "range": "stddev: 0.0000931300188197",
            "extra": "mean: 3.41413519118 msec\nrounds: 293"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[default-None]",
            "value": 444.882945757,
            "unit": "iter/sec",
            "range": "stddev: 0.000218789177763",
            "extra": "mean: 2.24778227517 msec\nrounds: 425"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[reduce-overhead-None]",
            "value": 442.635418637,
            "unit": "iter/sec",
            "range": "stddev: 0.000296806739976",
            "extra": "mean: 2.25919562217 msec\nrounds: 458"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[False-False-None]",
            "value": 507.541712258,
            "unit": "iter/sec",
            "range": "stddev: 0.0000284836998672",
            "extra": "mean: 1.97028140909 msec\nrounds: 494"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-False-None]",
            "value": 652.992131203,
            "unit": "iter/sec",
            "range": "stddev: 0.0000393620669349",
            "extra": "mean: 1.53141202201 msec\nrounds: 636"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[within-False-None]",
            "value": 198.368810465,
            "unit": "iter/sec",
            "range": "stddev: 0.000039163553254",
            "extra": "mean: 5.04111507074 msec\nrounds: 198"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-default-None]",
            "value": 114.39729397,
            "unit": "iter/sec",
            "range": "stddev: 0.000246713805403",
            "extra": "mean: 8.74146551285 msec\nrounds: 117"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[False-False-None]",
            "value": 139.474549771,
            "unit": "iter/sec",
            "range": "stddev: 0.000214196940264",
            "extra": "mean: 7.16976682587 msec\nrounds: 132"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[True-False-None]",
            "value": 264.78412323,
            "unit": "iter/sec",
            "range": "stddev: 0.0000404552318034",
            "extra": "mean: 3.77666148483 msec\nrounds: 262"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[within-False-None]",
            "value": 110.802209771,
            "unit": "iter/sec",
            "range": "stddev: 0.0000771956451659",
            "extra": "mean: 9.02509076365 msec\nrounds: 110"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 10.635202689,
            "unit": "iter/sec",
            "range": "stddev: 0.00124118295815",
            "extra": "mean: 94.0273569999 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_archive",
            "value": 4.394954406,
            "unit": "iter/sec",
            "range": "stddev: 0.0138572019004",
            "extra": "mean: 227.533646 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_zarr",
            "value": 0.410282934615,
            "unit": "iter/sec",
            "range": "stddev: 0.535725135141",
            "extra": "mean: 2.4373424182 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 40.8627025278,
            "unit": "iter/sec",
            "range": "stddev: 0.000637674639307",
            "extra": "mean: 24.4721944007 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.70174729366,
            "unit": "iter/sec",
            "range": "stddev: 0.0250245856468",
            "extra": "mean: 103.0742164 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 10.245837372,
            "unit": "iter/sec",
            "range": "stddev: 0.00138887893322",
            "extra": "mean: 97.6006121993 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 40.0814056573,
            "unit": "iter/sec",
            "range": "stddev: 0.00107069983891",
            "extra": "mean: 24.9492247989 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-compile]",
            "value": 4678.26067943,
            "unit": "iter/sec",
            "range": "stddev: 0.00000950528033975",
            "extra": "mean: 213.754655528 usec\nrounds: 1762"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-eager]",
            "value": 8220.20300597,
            "unit": "iter/sec",
            "range": "stddev: 0.00000799924519922",
            "extra": "mean: 121.651496839 usec\nrounds: 6882"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-compile]",
            "value": 6587.64872665,
            "unit": "iter/sec",
            "range": "stddev: 0.00000574123508637",
            "extra": "mean: 151.79922936 usec\nrounds: 3078"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-eager]",
            "value": 5850.30451685,
            "unit": "iter/sec",
            "range": "stddev: 0.000013903586445",
            "extra": "mean: 170.931273256 usec\nrounds: 5423"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-compile]",
            "value": 20740.5169378,
            "unit": "iter/sec",
            "range": "stddev: 0.00000369915123844",
            "extra": "mean: 48.214805976 usec\nrounds: 15828"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-eager]",
            "value": 30938.7902981,
            "unit": "iter/sec",
            "range": "stddev: 0.00000151994642256",
            "extra": "mean: 32.3218842872 usec\nrounds: 25707"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-compile]",
            "value": 63066.2716724,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021261513214",
            "extra": "mean: 15.8563360967 usec\nrounds: 37950"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-eager]",
            "value": 15760.7353519,
            "unit": "iter/sec",
            "range": "stddev: 0.0000114260383686",
            "extra": "mean: 63.4488161671 usec\nrounds: 15161"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-compile]",
            "value": 3635.13063015,
            "unit": "iter/sec",
            "range": "stddev: 0.0000106436847727",
            "extra": "mean: 275.093277723 usec\nrounds: 1289"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-eager]",
            "value": 5354.3919639,
            "unit": "iter/sec",
            "range": "stddev: 0.00000719274027161",
            "extra": "mean: 186.762569259 usec\nrounds: 4982"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-compile]",
            "value": 5862.68701448,
            "unit": "iter/sec",
            "range": "stddev: 0.00000703817830475",
            "extra": "mean: 170.570251751 usec\nrounds: 3241"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-eager]",
            "value": 16461.3883191,
            "unit": "iter/sec",
            "range": "stddev: 0.00000391795359945",
            "extra": "mean: 60.748217624 usec\nrounds: 14571"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-compile]",
            "value": 4088.46799162,
            "unit": "iter/sec",
            "range": "stddev: 0.00000809595083",
            "extra": "mean: 244.590394752 usec\nrounds: 1353"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-eager]",
            "value": 1969.17917107,
            "unit": "iter/sec",
            "range": "stddev: 0.0000185197042249",
            "extra": "mean: 507.825806149 usec\nrounds: 1878"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-eager]",
            "value": 4413.23931915,
            "unit": "iter/sec",
            "range": "stddev: 0.00000644137838804",
            "extra": "mean: 226.590929629 usec\nrounds: 4149"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-compile]",
            "value": 3676.24675962,
            "unit": "iter/sec",
            "range": "stddev: 0.0000063717658302",
            "extra": "mean: 272.016560745 usec\nrounds: 1316"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-eager]",
            "value": 13429.9904524,
            "unit": "iter/sec",
            "range": "stddev: 0.00000315648127855",
            "extra": "mean: 74.4602167474 usec\nrounds: 11520"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-compile]",
            "value": 5639.14242385,
            "unit": "iter/sec",
            "range": "stddev: 0.00000746563640521",
            "extra": "mean: 177.331928304 usec\nrounds: 3306"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-eager]",
            "value": 2216.2508116,
            "unit": "iter/sec",
            "range": "stddev: 0.0000247172677046",
            "extra": "mean: 451.212468718 usec\nrounds: 2204"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-compile]",
            "value": 4108.08667754,
            "unit": "iter/sec",
            "range": "stddev: 0.00000713213347414",
            "extra": "mean: 243.422322481 usec\nrounds: 1349"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-compile]",
            "value": 33728.5052216,
            "unit": "iter/sec",
            "range": "stddev: 0.00000285814496671",
            "extra": "mean: 29.648512243 usec\nrounds: 21418"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-eager]",
            "value": 53579.4169863,
            "unit": "iter/sec",
            "range": "stddev: 0.00000116805097766",
            "extra": "mean: 18.6638835629 usec\nrounds: 45066"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-compile]",
            "value": 46478.9658149,
            "unit": "iter/sec",
            "range": "stddev: 0.00000208925283401",
            "extra": "mean: 21.5151086619 usec\nrounds: 27307"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-eager]",
            "value": 19548.448964,
            "unit": "iter/sec",
            "range": "stddev: 0.0000117695820723",
            "extra": "mean: 51.1549536152 usec\nrounds: 17289"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-compile]",
            "value": 3900.82606421,
            "unit": "iter/sec",
            "range": "stddev: 0.000228603186816",
            "extra": "mean: 256.355957313 usec\nrounds: 94"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-eager]",
            "value": 316.28958094,
            "unit": "iter/sec",
            "range": "stddev: 0.0000317724001567",
            "extra": "mean: 3.16165963175 msec\nrounds: 315"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-compile]",
            "value": 5063.25040688,
            "unit": "iter/sec",
            "range": "stddev: 0.000212038502416",
            "extra": "mean: 197.501588829 usec\nrounds: 90"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-eager]",
            "value": 354.751106893,
            "unit": "iter/sec",
            "range": "stddev: 0.0000781824171005",
            "extra": "mean: 2.81887774434 msec\nrounds: 356"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-compile]",
            "value": 6034.23137835,
            "unit": "iter/sec",
            "range": "stddev: 0.00000881895457962",
            "extra": "mean: 165.721189212 usec\nrounds: 3789"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-eager]",
            "value": 18204.8684808,
            "unit": "iter/sec",
            "range": "stddev: 0.0000127905983071",
            "extra": "mean: 54.9303611314 usec\nrounds: 15092"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-compile]",
            "value": 7104.96467011,
            "unit": "iter/sec",
            "range": "stddev: 0.0000156383377804",
            "extra": "mean: 140.746653422 usec\nrounds: 4386"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-eager]",
            "value": 21362.4135719,
            "unit": "iter/sec",
            "range": "stddev: 0.0000126550418079",
            "extra": "mean: 46.8111899732 usec\nrounds: 17989"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-compile]",
            "value": 7244.26483665,
            "unit": "iter/sec",
            "range": "stddev: 0.00000675077809183",
            "extra": "mean: 138.040232177 usec\nrounds: 6628"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-eager]",
            "value": 21594.4588084,
            "unit": "iter/sec",
            "range": "stddev: 0.0000120852440471",
            "extra": "mean: 46.3081760406 usec\nrounds: 18488"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-compile]",
            "value": 11322.536843,
            "unit": "iter/sec",
            "range": "stddev: 0.00000532838705214",
            "extra": "mean: 88.319429989 usec\nrounds: 9349"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-eager]",
            "value": 39098.3484711,
            "unit": "iter/sec",
            "range": "stddev: 0.0000117865011079",
            "extra": "mean: 25.5765278868 usec\nrounds: 33367"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-compile]",
            "value": 15316.1765125,
            "unit": "iter/sec",
            "range": "stddev: 0.00000420031255836",
            "extra": "mean: 65.2904462928 usec\nrounds: 12763"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-eager]",
            "value": 47833.4039155,
            "unit": "iter/sec",
            "range": "stddev: 0.0000105041993837",
            "extra": "mean: 20.9058924965 usec\nrounds: 36063"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-compile]",
            "value": 15106.9488266,
            "unit": "iter/sec",
            "range": "stddev: 0.00000427950500129",
            "extra": "mean: 66.1947036081 usec\nrounds: 13357"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-eager]",
            "value": 47710.2304315,
            "unit": "iter/sec",
            "range": "stddev: 0.0000105731344966",
            "extra": "mean: 20.9598652314 usec\nrounds: 35050"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-compile]",
            "value": 11239.3911761,
            "unit": "iter/sec",
            "range": "stddev: 0.00000469298261968",
            "extra": "mean: 88.9727908149 usec\nrounds: 9981"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-eager]",
            "value": 47238.0050715,
            "unit": "iter/sec",
            "range": "stddev: 0.0000106979640438",
            "extra": "mean: 21.1693952462 usec\nrounds: 37370"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-compile]",
            "value": 15303.5079001,
            "unit": "iter/sec",
            "range": "stddev: 0.00000485814413527",
            "extra": "mean: 65.3444952965 usec\nrounds: 13398"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-eager]",
            "value": 47288.9830457,
            "unit": "iter/sec",
            "range": "stddev: 0.0000104536528199",
            "extra": "mean: 21.1465744365 usec\nrounds: 36969"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-compile]",
            "value": 15396.6887635,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036660649417",
            "extra": "mean: 64.9490299739 usec\nrounds: 13485"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-eager]",
            "value": 48129.2792338,
            "unit": "iter/sec",
            "range": "stddev: 0.000010489232566",
            "extra": "mean: 20.777373273 usec\nrounds: 36087"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-eager]",
            "value": 33049.6097308,
            "unit": "iter/sec",
            "range": "stddev: 0.00000256782843941",
            "extra": "mean: 30.2575433763 usec\nrounds: 24420"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-compile]",
            "value": 6698.9811482,
            "unit": "iter/sec",
            "range": "stddev: 0.0000127567813457",
            "extra": "mean: 149.276431427 usec\nrounds: 4300"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-eager]",
            "value": 2661.34555954,
            "unit": "iter/sec",
            "range": "stddev: 0.0000145413467333",
            "extra": "mean: 375.749776806 usec\nrounds: 2666"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-compile]",
            "value": 5993.20720519,
            "unit": "iter/sec",
            "range": "stddev: 0.0000135037671541",
            "extra": "mean: 166.855569274 usec\nrounds: 3436"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[eager]",
            "value": 6071.17694152,
            "unit": "iter/sec",
            "range": "stddev: 0.0000123028391928",
            "extra": "mean: 164.712708859 usec\nrounds: 4682"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[compile]",
            "value": 5518.14093686,
            "unit": "iter/sec",
            "range": "stddev: 0.00000787944636136",
            "extra": "mean: 181.220452947 usec\nrounds: 3751"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[eager]",
            "value": 5830.11684462,
            "unit": "iter/sec",
            "range": "stddev: 0.00000595515868701",
            "extra": "mean: 171.523148961 usec\nrounds: 5647"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[compile]",
            "value": 5160.89812729,
            "unit": "iter/sec",
            "range": "stddev: 0.0000120346656366",
            "extra": "mean: 193.764723763 usec\nrounds: 3558"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-eager]",
            "value": 108180.96754,
            "unit": "iter/sec",
            "range": "stddev: 0.00000187945096307",
            "extra": "mean: 9.24377016348 usec\nrounds: 85463"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-compile]",
            "value": 39767.5952647,
            "unit": "iter/sec",
            "range": "stddev: 0.00000349465877299",
            "extra": "mean: 25.1461018285 usec\nrounds: 26617"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-eager]",
            "value": 66184.3436748,
            "unit": "iter/sec",
            "range": "stddev: 0.00000148325955103",
            "extra": "mean: 15.1093135397 usec\nrounds: 55158"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-compile]",
            "value": 32465.3020159,
            "unit": "iter/sec",
            "range": "stddev: 0.00000290064084654",
            "extra": "mean: 30.8021160409 usec\nrounds: 23441"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-eager]",
            "value": 37213.5519219,
            "unit": "iter/sec",
            "range": "stddev: 0.00000183969715791",
            "extra": "mean: 26.8719310132 usec\nrounds: 33377"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-compile]",
            "value": 24095.7748887,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027173386875",
            "extra": "mean: 41.5010517246 usec\nrounds: 18015"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[eager]",
            "value": 36835.3706281,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024993756713",
            "extra": "mean: 27.1478196893 usec\nrounds: 24790"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[compile]",
            "value": 8524.84245958,
            "unit": "iter/sec",
            "range": "stddev: 0.00000888605616822",
            "extra": "mean: 117.304220546 usec\nrounds: 7118"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[eager]",
            "value": 31396.1965117,
            "unit": "iter/sec",
            "range": "stddev: 0.00000463838259806",
            "extra": "mean: 31.8509918749 usec\nrounds: 21322"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile]",
            "value": 8427.19119897,
            "unit": "iter/sec",
            "range": "stddev: 0.0000240044913749",
            "extra": "mean: 118.663499663 usec\nrounds: 5722"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile-overhead]",
            "value": 5300.8817487,
            "unit": "iter/sec",
            "range": "stddev: 0.0000109255899074",
            "extra": "mean: 188.647860376 usec\nrounds: 308"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[eager]",
            "value": 3743.00587104,
            "unit": "iter/sec",
            "range": "stddev: 0.0000101462898954",
            "extra": "mean: 267.164956309 usec\nrounds: 3547"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile]",
            "value": 3149.68626558,
            "unit": "iter/sec",
            "range": "stddev: 0.00000880672844438",
            "extra": "mean: 317.491939095 usec\nrounds: 2477"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile-overhead]",
            "value": 255.24168001,
            "unit": "iter/sec",
            "range": "stddev: 0.00173639029145",
            "extra": "mean: 3.91785542221 msec\nrounds: 964"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[False]",
            "value": 6183.11724091,
            "unit": "iter/sec",
            "range": "stddev: 0.00000695314437438",
            "extra": "mean: 161.73071948 usec\nrounds: 5219"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[True]",
            "value": 6185.64929616,
            "unit": "iter/sec",
            "range": "stddev: 0.00000759212421479",
            "extra": "mean: 161.664516063 usec\nrounds: 5292"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[eager]",
            "value": 641.400251594,
            "unit": "iter/sec",
            "range": "stddev: 0.0000236280046067",
            "extra": "mean: 1.55908888017 msec\nrounds: 617"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile]",
            "value": 694.543314026,
            "unit": "iter/sec",
            "range": "stddev: 0.0000417968446373",
            "extra": "mean: 1.43979501322 msec\nrounds: 379"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile-overhead]",
            "value": 1072.29342556,
            "unit": "iter/sec",
            "range": "stddev: 0.0000789947272922",
            "extra": "mean: 932.58055693 usec\nrounds: 88"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[eager]",
            "value": 10254.9099006,
            "unit": "iter/sec",
            "range": "stddev: 0.00000570128977691",
            "extra": "mean: 97.5142648444 usec\nrounds: 8748"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile]",
            "value": 7886.92412181,
            "unit": "iter/sec",
            "range": "stddev: 0.00000529566350745",
            "extra": "mean: 126.792141593 usec\nrounds: 6716"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile-overhead]",
            "value": 4912.92650664,
            "unit": "iter/sec",
            "range": "stddev: 0.00000630634570315",
            "extra": "mean: 203.544669078 usec\nrounds: 3303"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[eager]",
            "value": 2539.86982355,
            "unit": "iter/sec",
            "range": "stddev: 0.0000123183385756",
            "extra": "mean: 393.720965826 usec\nrounds: 2428"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile]",
            "value": 2916.09568338,
            "unit": "iter/sec",
            "range": "stddev: 0.00000724896874391",
            "extra": "mean: 342.924275668 usec\nrounds: 2717"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile-overhead]",
            "value": 3251.69774692,
            "unit": "iter/sec",
            "range": "stddev: 0.00000711262317334",
            "extra": "mean: 307.531658176 usec\nrounds: 2077"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-eager]",
            "value": 1211.70096263,
            "unit": "iter/sec",
            "range": "stddev: 0.000028640643141",
            "extra": "mean: 825.286131513 usec\nrounds: 1217"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile]",
            "value": 1423.6295717,
            "unit": "iter/sec",
            "range": "stddev: 0.0000130447271499",
            "extra": "mean: 702.429915673 usec\nrounds: 1375"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile-overhead]",
            "value": 2120.98635678,
            "unit": "iter/sec",
            "range": "stddev: 0.0000111275787232",
            "extra": "mean: 471.478751764 usec\nrounds: 1297"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-eager]",
            "value": 791.402818273,
            "unit": "iter/sec",
            "range": "stddev: 0.0000211395416725",
            "extra": "mean: 1.26357902311 msec\nrounds: 780"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile]",
            "value": 1321.30330313,
            "unit": "iter/sec",
            "range": "stddev: 0.0000215487357843",
            "extra": "mean: 756.828502304 usec\nrounds: 1304"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile-overhead]",
            "value": 1995.08555849,
            "unit": "iter/sec",
            "range": "stddev: 0.0000184821553788",
            "extra": "mean: 501.231636779 usec\nrounds: 1148"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-eager]",
            "value": 1271.73242083,
            "unit": "iter/sec",
            "range": "stddev: 0.0000154624399215",
            "extra": "mean: 786.328934942 usec\nrounds: 1244"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile]",
            "value": 1390.16068741,
            "unit": "iter/sec",
            "range": "stddev: 0.0000226175437566",
            "extra": "mean: 719.341302811 usec\nrounds: 1364"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile-overhead]",
            "value": 2123.24378545,
            "unit": "iter/sec",
            "range": "stddev: 0.0000268313802676",
            "extra": "mean: 470.977476468 usec\nrounds: 1066"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-eager]",
            "value": 758.759031391,
            "unit": "iter/sec",
            "range": "stddev: 0.0000276136260669",
            "extra": "mean: 1.31794147895 msec\nrounds: 737"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile]",
            "value": 1207.57368895,
            "unit": "iter/sec",
            "range": "stddev: 0.0000250364111069",
            "extra": "mean: 828.106813814 usec\nrounds: 1117"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile-overhead]",
            "value": 1334.9337835,
            "unit": "iter/sec",
            "range": "stddev: 0.0000454955619428",
            "extra": "mean: 749.100826092 usec\nrounds: 1035"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[eager]",
            "value": 333.183441948,
            "unit": "iter/sec",
            "range": "stddev: 0.00016836422718",
            "extra": "mean: 3.00134962936 msec\nrounds: 348"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile]",
            "value": 1335.53165592,
            "unit": "iter/sec",
            "range": "stddev: 0.0000115809436603",
            "extra": "mean: 748.765478954 usec\nrounds: 1019"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile-overhead]",
            "value": 1930.90102227,
            "unit": "iter/sec",
            "range": "stddev: 0.0000354381435783",
            "extra": "mean: 517.892936235 usec\nrounds: 126"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1057.02950274,
            "unit": "iter/sec",
            "range": "stddev: 0.0000380042626603",
            "extra": "mean: 946.047387901 usec\nrounds: 972"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1056.15713908,
            "unit": "iter/sec",
            "range": "stddev: 0.0000391888840844",
            "extra": "mean: 946.828803211 usec\nrounds: 1062"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1144.4359545,
            "unit": "iter/sec",
            "range": "stddev: 0.0000397203283688",
            "extra": "mean: 873.792889913 usec\nrounds: 1135"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1136.47386003,
            "unit": "iter/sec",
            "range": "stddev: 0.0000475513315481",
            "extra": "mean: 879.914651068 usec\nrounds: 1155"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-True]",
            "value": 37.8733723866,
            "unit": "iter/sec",
            "range": "stddev: 0.000394269815437",
            "extra": "mean: 26.4037749211 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-False]",
            "value": 38.3860695936,
            "unit": "iter/sec",
            "range": "stddev: 0.000126351508208",
            "extra": "mean: 26.0511172565 msec\nrounds: 39"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-True]",
            "value": 38.127203522,
            "unit": "iter/sec",
            "range": "stddev: 0.000397847172438",
            "extra": "mean: 26.2279922896 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-False]",
            "value": 37.8144858757,
            "unit": "iter/sec",
            "range": "stddev: 0.000424407674537",
            "extra": "mean: 26.4448921317 msec\nrounds: 38"
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
          "id": "8129b59aabb2b14c0a554fb04019d05b5608c09f",
          "message": "[BugFix] Keep the fields of later records in from_json without pandas (#3065)",
          "timestamp": "2026-10-20T02:42:14-04:00",
          "tree_id": "52469a1c3e6317a9bdceb0d7695d2917232176e7",
          "url": "https://github.com/pytorch/tensordict/commit/8129b59aabb2b14c0a554fb04019d05b5608c09f"
        },
        "date": 1792479286262,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-2]",
            "value": 3186.06601541,
            "unit": "iter/sec",
            "range": "stddev: 0.0000637489932679",
            "extra": "mean: 313.866691765 usec\nrounds: 120"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-8]",
            "value": 3079.16438754,
            "unit": "iter/sec",
            "range": "stddev: 0.000058823205262",
            "extra": "mean: 324.763433887 usec\nrounds: 3238"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-32]",
            "value": 1690.0917698,
            "unit": "iter/sec",
            "range": "stddev: 0.000139439071697",
            "extra": "mean: 591.683846919 usec\nrounds: 1777"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-2]",
            "value": 4394.55647962,
            "unit": "iter/sec",
            "range": "stddev: 0.0000130757506061",
            "extra": "mean: 227.554249135 usec\nrounds: 3998"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-8]",
            "value": 3616.51992144,
            "unit": "iter/sec",
            "range": "stddev: 0.00000943041496296",
            "extra": "mean: 276.508915123 usec\nrounds: 3240"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-32]",
            "value": 1795.9330492,
            "unit": "iter/sec",
            "range": "stddev: 0.0000257562148161",
            "extra": "mean: 556.8136298 usec\nrounds: 1664"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-2]",
            "value": 2245.42944367,
            "unit": "iter/sec",
            "range": "stddev: 0.0000429000981534",
            "extra": "mean: 445.349108083 usec\nrounds: 74"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-8]",
            "value": 614.39131485,
            "unit": "iter/sec",
            "range": "stddev: 0.000052008818216",
            "extra": "mean: 1.62762717478 msec\nrounds: 595"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-32]",
            "value": 157.150777801,
            "unit": "iter/sec",
            "range": "stddev: 0.000052013468454",
            "extra": "mean: 6.3633156259 msec\nrounds: 155"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-2]",
            "value": 2319.87017523,
            "unit": "iter/sec",
            "range": "stddev: 0.0000134017908668",
            "extra": "mean: 431.058604347 usec\nrounds: 2123"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-8]",
            "value": 617.866965531,
            "unit": "iter/sec",
            "range": "stddev: 0.0000224357558175",
            "extra": "mean: 1.61847137942 msec\nrounds: 601"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-32]",
            "value": 146.530778972,
            "unit": "iter/sec",
            "range": "stddev: 0.000129764815548",
            "extra": "mean: 6.82450476971 msec\nrounds: 152"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-2]",
            "value": 3018.34636764,
            "unit": "iter/sec",
            "range": "stddev: 0.0000127137002779",
            "extra": "mean: 331.307238534 usec\nrounds: 2398"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-8]",
            "value": 845.095373416,
            "unit": "iter/sec",
            "range": "stddev: 0.0000218286098077",
            "extra": "mean: 1.1832983962 msec\nrounds: 795"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-32]",
            "value": 216.983760793,
            "unit": "iter/sec",
            "range": "stddev: 0.0000343532375309",
            "extra": "mean: 4.60863981868 msec\nrounds: 215"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-2]",
            "value": 3136.50579363,
            "unit": "iter/sec",
            "range": "stddev: 0.00000937677105419",
            "extra": "mean: 318.826128754 usec\nrounds: 2943"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-8]",
            "value": 841.702587729,
            "unit": "iter/sec",
            "range": "stddev: 0.0000236262678567",
            "extra": "mean: 1.18806810693 msec\nrounds: 823"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-32]",
            "value": 205.700576403,
            "unit": "iter/sec",
            "range": "stddev: 0.000361773488057",
            "extra": "mean: 4.86143508923 msec\nrounds: 213"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-2]",
            "value": 3383.27505553,
            "unit": "iter/sec",
            "range": "stddev: 0.00000921636006599",
            "extra": "mean: 295.571593674 usec\nrounds: 2264"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-8]",
            "value": 906.60804632,
            "unit": "iter/sec",
            "range": "stddev: 0.0000200576540085",
            "extra": "mean: 1.10301249152 msec\nrounds: 877"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-32]",
            "value": 232.185132082,
            "unit": "iter/sec",
            "range": "stddev: 0.0000634917653498",
            "extra": "mean: 4.30690798775 msec\nrounds: 248"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-2]",
            "value": 3372.45063611,
            "unit": "iter/sec",
            "range": "stddev: 0.0000179516321174",
            "extra": "mean: 296.520277952 usec\nrounds: 3162"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-8]",
            "value": 925.716531375,
            "unit": "iter/sec",
            "range": "stddev: 0.0000418119298207",
            "extra": "mean: 1.08024429305 msec\nrounds: 833"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-32]",
            "value": 250.380811083,
            "unit": "iter/sec",
            "range": "stddev: 0.0000285109644206",
            "extra": "mean: 3.99391628966 msec\nrounds: 245"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-128]",
            "value": 189.647806889,
            "unit": "iter/sec",
            "range": "stddev: 0.0000268605069139",
            "extra": "mean: 5.27293205444 msec\nrounds: 184"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-256]",
            "value": 99.6686524995,
            "unit": "iter/sec",
            "range": "stddev: 0.000102305019097",
            "extra": "mean: 10.0332449062 msec\nrounds: 96"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-128]",
            "value": 181.444232132,
            "unit": "iter/sec",
            "range": "stddev: 0.0000866035238455",
            "extra": "mean: 5.51133529156 msec\nrounds: 175"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-256]",
            "value": 95.1056159134,
            "unit": "iter/sec",
            "range": "stddev: 0.000105640826952",
            "extra": "mean: 10.5146261911 msec\nrounds: 89"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-128]",
            "value": 12.3272758787,
            "unit": "iter/sec",
            "range": "stddev: 0.000636883438464",
            "extra": "mean: 81.1209232148 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-256]",
            "value": 6.61564572313,
            "unit": "iter/sec",
            "range": "stddev: 0.00125362841591",
            "extra": "mean: 151.156824572 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-128]",
            "value": 12.309197947,
            "unit": "iter/sec",
            "range": "stddev: 0.00145492111345",
            "extra": "mean: 81.2400616435 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-256]",
            "value": 6.49196150905,
            "unit": "iter/sec",
            "range": "stddev: 0.000592250599668",
            "extra": "mean: 154.036649571 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-128]",
            "value": 23.8811589233,
            "unit": "iter/sec",
            "range": "stddev: 0.000113368211423",
            "extra": "mean: 41.8740147081 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-256]",
            "value": 11.5250317403,
            "unit": "iter/sec",
            "range": "stddev: 0.00242455972139",
            "extra": "mean: 86.7676569172 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-128]",
            "value": 23.346234122,
            "unit": "iter/sec",
            "range": "stddev: 0.00104227181792",
            "extra": "mean: 42.8334606248 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-256]",
            "value": 11.9036800042,
            "unit": "iter/sec",
            "range": "stddev: 0.000628779313136",
            "extra": "mean: 84.0076345839 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-128]",
            "value": 27.8683593179,
            "unit": "iter/sec",
            "range": "stddev: 0.000197226524779",
            "extra": "mean: 35.8829878929 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-256]",
            "value": 14.0140223467,
            "unit": "iter/sec",
            "range": "stddev: 0.000299321306641",
            "extra": "mean: 71.3571004287 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-128]",
            "value": 27.8983701432,
            "unit": "iter/sec",
            "range": "stddev: 0.000219777177401",
            "extra": "mean: 35.8443878573 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-256]",
            "value": 13.8902602076,
            "unit": "iter/sec",
            "range": "stddev: 0.000536869103014",
            "extra": "mean: 71.9928917858 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_mask_many_leaves[cuda]",
            "value": 6521.60394733,
            "unit": "iter/sec",
            "range": "stddev: 0.0000152484448124",
            "extra": "mean: 153.336511704 usec\nrounds: 43"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-None]",
            "value": 6868.92748209,
            "unit": "iter/sec",
            "range": "stddev: 0.00000406885238109",
            "extra": "mean: 145.583135447 usec\nrounds: 1448"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-cpu]",
            "value": 6647.38434548,
            "unit": "iter/sec",
            "range": "stddev: 0.0000046666494942",
            "extra": "mean: 150.435110718 usec\nrounds: 4940"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to",
            "value": 22714.7775354,
            "unit": "iter/sec",
            "range": "stddev: 0.00000297642608521",
            "extra": "mean: 44.0242039986 usec\nrounds: 11358"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_nonblocking",
            "value": 23975.7870333,
            "unit": "iter/sec",
            "range": "stddev: 0.00000217545904709",
            "extra": "mean: 41.7087455194 usec\nrounds: 12528"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[add]",
            "value": 10739.8203593,
            "unit": "iter/sec",
            "range": "stddev: 0.00000696589902501",
            "extra": "mean: 93.1114270581 usec\nrounds: 192"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[sub]",
            "value": 11009.4078431,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032929312045",
            "extra": "mean: 90.83140658 usec\nrounds: 8213"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_blocking",
            "value": 57.5224793252,
            "unit": "iter/sec",
            "range": "stddev: 0.0000901803728489",
            "extra": "mean: 17.3845079651 msec\nrounds: 58"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_global_sync",
            "value": 96.3377163609,
            "unit": "iter/sec",
            "range": "stddev: 0.01179773024",
            "extra": "mean: 10.3801505555 msec\nrounds: 108"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_event_sync",
            "value": 98.3330279206,
            "unit": "iter/sec",
            "range": "stddev: 0.000155881889228",
            "extra": "mean: 10.1695231109 msec\nrounds: 99"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_default",
            "value": 96.4511318908,
            "unit": "iter/sec",
            "range": "stddev: 0.000106375755509",
            "extra": "mean: 10.3679446824 msec\nrounds: 85"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_cpu_to_cpu_default",
            "value": 3890.57076098,
            "unit": "iter/sec",
            "range": "stddev: 0.00000568566585529",
            "extra": "mean: 257.031695716 usec\nrounds: 3931"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[False-None]",
            "value": 298.610282877,
            "unit": "iter/sec",
            "range": "stddev: 0.000021475519328",
            "extra": "mean: 3.34884649774 msec\nrounds: 233"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[default-None]",
            "value": 449.091267795,
            "unit": "iter/sec",
            "range": "stddev: 0.000234759130978",
            "extra": "mean: 2.22671886922 msec\nrounds: 428"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[reduce-overhead-None]",
            "value": 435.234006897,
            "unit": "iter/sec",
            "range": "stddev: 0.000325831340075",
            "extra": "mean: 2.29761458009 msec\nrounds: 431"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[False-False-None]",
            "value": 485.55666531,
            "unit": "iter/sec",
            "range": "stddev: 0.0000414729934762",
            "extra": "mean: 2.05949186047 msec\nrounds: 444"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-False-None]",
            "value": 628.628958933,
            "unit": "iter/sec",
            "range": "stddev: 0.0000730990977661",
            "extra": "mean: 1.59076349536 msec\nrounds: 636"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[within-False-None]",
            "value": 186.884700259,
            "unit": "iter/sec",
            "range": "stddev: 0.000233518812281",
            "extra": "mean: 5.3508928158 msec\nrounds: 190"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-default-None]",
            "value": 112.363130479,
            "unit": "iter/sec",
            "range": "stddev: 0.000319206934109",
            "extra": "mean: 8.89971644381 msec\nrounds: 115"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[False-False-None]",
            "value": 139.449489795,
            "unit": "iter/sec",
            "range": "stddev: 0.000147251163282",
            "extra": "mean: 7.17105527939 msec\nrounds: 136"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[True-False-None]",
            "value": 262.134173444,
            "unit": "iter/sec",
            "range": "stddev: 0.000104990185191",
            "extra": "mean: 3.81484026619 msec\nrounds: 263"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[within-False-None]",
            "value": 110.651102753,
            "unit": "iter/sec",
            "range": "stddev: 0.000294316236977",
            "extra": "mean: 9.03741558032 msec\nrounds: 112"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 9.51455445403,
            "unit": "iter/sec",
            "range": "stddev: 0.0176578012148",
            "extra": "mean: 105.102136399 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_archive",
            "value": 4.5690798785,
            "unit": "iter/sec",
            "range": "stddev: 0.0016997733878",
            "extra": "mean: 218.862446399 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_zarr",
            "value": 0.419737181544,
            "unit": "iter/sec",
            "range": "stddev: 0.0300352911846",
            "extra": "mean: 2.3824432144 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 39.9962525112,
            "unit": "iter/sec",
            "range": "stddev: 0.000251157923411",
            "extra": "mean: 25.0023424 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 10.6111043662,
            "unit": "iter/sec",
            "range": "stddev: 0.000841141162194",
            "extra": "mean: 94.2408976 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 9.4413615127,
            "unit": "iter/sec",
            "range": "stddev: 0.00787076117425",
            "extra": "mean: 105.916927199 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 41.8750347455,
            "unit": "iter/sec",
            "range": "stddev: 0.000323147636086",
            "extra": "mean: 23.8805772002 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-compile]",
            "value": 4838.18984725,
            "unit": "iter/sec",
            "range": "stddev: 0.00000852646050109",
            "extra": "mean: 206.688871576 usec\nrounds: 1713"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-eager]",
            "value": 8112.88941429,
            "unit": "iter/sec",
            "range": "stddev: 0.00000723973905696",
            "extra": "mean: 123.260647215 usec\nrounds: 6792"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-compile]",
            "value": 6601.19477418,
            "unit": "iter/sec",
            "range": "stddev: 0.00000804232372967",
            "extra": "mean: 151.487728239 usec\nrounds: 3065"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-eager]",
            "value": 5731.15166257,
            "unit": "iter/sec",
            "range": "stddev: 0.0000158584187724",
            "extra": "mean: 174.485000376 usec\nrounds: 5424"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-compile]",
            "value": 20307.3917534,
            "unit": "iter/sec",
            "range": "stddev: 0.00000359984066697",
            "extra": "mean: 49.2431530421 usec\nrounds: 16067"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-eager]",
            "value": 31571.6726378,
            "unit": "iter/sec",
            "range": "stddev: 0.00000195112153282",
            "extra": "mean: 31.6739632859 usec\nrounds: 26123"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-compile]",
            "value": 46319.6412636,
            "unit": "iter/sec",
            "range": "stddev: 0.00000424426373645",
            "extra": "mean: 21.5891136615 usec\nrounds: 38109"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-eager]",
            "value": 16109.2517694,
            "unit": "iter/sec",
            "range": "stddev: 0.0000120224896486",
            "extra": "mean: 62.0761295627 usec\nrounds: 15404"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-compile]",
            "value": 3777.58901282,
            "unit": "iter/sec",
            "range": "stddev: 0.0000083947022433",
            "extra": "mean: 264.719109624 usec\nrounds: 1323"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-eager]",
            "value": 5124.32922686,
            "unit": "iter/sec",
            "range": "stddev: 0.0000332391918635",
            "extra": "mean: 195.147492624 usec\nrounds: 5085"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-compile]",
            "value": 6050.17125307,
            "unit": "iter/sec",
            "range": "stddev: 0.00000662653672231",
            "extra": "mean: 165.284577605 usec\nrounds: 3343"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-eager]",
            "value": 15312.5892637,
            "unit": "iter/sec",
            "range": "stddev: 0.0000162071261917",
            "extra": "mean: 65.3057417513 usec\nrounds: 13998"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-compile]",
            "value": 4182.96920955,
            "unit": "iter/sec",
            "range": "stddev: 0.00000790851555011",
            "extra": "mean: 239.064633255 usec\nrounds: 1399"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-eager]",
            "value": 1863.90520797,
            "unit": "iter/sec",
            "range": "stddev: 0.0000879341764565",
            "extra": "mean: 536.507970323 usec\nrounds: 1919"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-eager]",
            "value": 4375.21661777,
            "unit": "iter/sec",
            "range": "stddev: 0.00000849922802653",
            "extra": "mean: 228.560111958 usec\nrounds: 4075"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-compile]",
            "value": 3559.8789657,
            "unit": "iter/sec",
            "range": "stddev: 0.0000102793762791",
            "extra": "mean: 280.908426841 usec\nrounds: 1305"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-eager]",
            "value": 12910.4472176,
            "unit": "iter/sec",
            "range": "stddev: 0.00000397955629763",
            "extra": "mean: 77.4566506601 usec\nrounds: 11416"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-compile]",
            "value": 5702.44827985,
            "unit": "iter/sec",
            "range": "stddev: 0.0000200973160373",
            "extra": "mean: 175.36327397 usec\nrounds: 2515"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-eager]",
            "value": 2126.71968092,
            "unit": "iter/sec",
            "range": "stddev: 0.0000552016348146",
            "extra": "mean: 470.207714242 usec\nrounds: 2002"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-compile]",
            "value": 4056.00200054,
            "unit": "iter/sec",
            "range": "stddev: 0.0000150315554403",
            "extra": "mean: 246.548201865 usec\nrounds: 1367"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-compile]",
            "value": 31505.3699086,
            "unit": "iter/sec",
            "range": "stddev: 0.00000309072247473",
            "extra": "mean: 31.740620818 usec\nrounds: 22119"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-eager]",
            "value": 54660.2849873,
            "unit": "iter/sec",
            "range": "stddev: 0.00000196237246409",
            "extra": "mean: 18.2948186281 usec\nrounds: 46105"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-compile]",
            "value": 44588.670903,
            "unit": "iter/sec",
            "range": "stddev: 0.00000204367114091",
            "extra": "mean: 22.4272215284 usec\nrounds: 28711"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-eager]",
            "value": 20042.1535704,
            "unit": "iter/sec",
            "range": "stddev: 0.0000114062763975",
            "extra": "mean: 49.8948377222 usec\nrounds: 16835"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-compile]",
            "value": 4201.85974045,
            "unit": "iter/sec",
            "range": "stddev: 0.000209359914502",
            "extra": "mean: 237.989857294 usec\nrounds: 98"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-eager]",
            "value": 302.970548622,
            "unit": "iter/sec",
            "range": "stddev: 0.0000541333776447",
            "extra": "mean: 3.30065085385 msec\nrounds: 301"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-compile]",
            "value": 5011.97205127,
            "unit": "iter/sec",
            "range": "stddev: 0.000227158299304",
            "extra": "mean: 199.52226185 usec\nrounds: 84"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-eager]",
            "value": 357.74057578,
            "unit": "iter/sec",
            "range": "stddev: 0.0000497248105614",
            "extra": "mean: 2.79532171552 msec\nrounds: 348"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-compile]",
            "value": 6302.00549097,
            "unit": "iter/sec",
            "range": "stddev: 0.00000470667588678",
            "extra": "mean: 158.679645937 usec\nrounds: 3327"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-eager]",
            "value": 19173.8903196,
            "unit": "iter/sec",
            "range": "stddev: 0.0000116485953314",
            "extra": "mean: 52.1542568216 usec\nrounds: 15731"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-compile]",
            "value": 7794.00583926,
            "unit": "iter/sec",
            "range": "stddev: 0.00000510537299072",
            "extra": "mean: 128.303727329 usec\nrounds: 4368"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-eager]",
            "value": 22355.8815403,
            "unit": "iter/sec",
            "range": "stddev: 0.0000115845564864",
            "extra": "mean: 44.730958079 usec\nrounds: 18450"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-compile]",
            "value": 7691.3144908,
            "unit": "iter/sec",
            "range": "stddev: 0.00000479010501398",
            "extra": "mean: 130.016787273 usec\nrounds: 6948"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-eager]",
            "value": 22363.2882656,
            "unit": "iter/sec",
            "range": "stddev: 0.0000115923228601",
            "extra": "mean: 44.7161431773 usec\nrounds: 18543"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-compile]",
            "value": 11831.4993991,
            "unit": "iter/sec",
            "range": "stddev: 0.00000459661956967",
            "extra": "mean: 84.5201412154 usec\nrounds: 9695"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-eager]",
            "value": 45566.9573509,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113765432998",
            "extra": "mean: 21.945726863 usec\nrounds: 34059"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-compile]",
            "value": 15964.2004444,
            "unit": "iter/sec",
            "range": "stddev: 0.00000574854633793",
            "extra": "mean: 62.6401556085 usec\nrounds: 12981"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-eager]",
            "value": 47681.1249383,
            "unit": "iter/sec",
            "range": "stddev: 0.0000109327960542",
            "extra": "mean: 20.9726595439 usec\nrounds: 36389"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-compile]",
            "value": 15395.3110361,
            "unit": "iter/sec",
            "range": "stddev: 0.00000341512673833",
            "extra": "mean: 64.9548422671 usec\nrounds: 13382"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-eager]",
            "value": 46141.0580187,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113563764487",
            "extra": "mean: 21.6726716495 usec\nrounds: 34709"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-compile]",
            "value": 11349.0050002,
            "unit": "iter/sec",
            "range": "stddev: 0.00000529065175282",
            "extra": "mean: 88.1134513541 usec\nrounds: 10070"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-eager]",
            "value": 43279.410181,
            "unit": "iter/sec",
            "range": "stddev: 0.0000110872629367",
            "extra": "mean: 23.1056753273 usec\nrounds: 35173"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-compile]",
            "value": 15445.4566811,
            "unit": "iter/sec",
            "range": "stddev: 0.00000434536379628",
            "extra": "mean: 64.7439580871 usec\nrounds: 13412"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-eager]",
            "value": 40636.3275333,
            "unit": "iter/sec",
            "range": "stddev: 0.0000123564917687",
            "extra": "mean: 24.6085229818 usec\nrounds: 35087"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-compile]",
            "value": 15698.2985112,
            "unit": "iter/sec",
            "range": "stddev: 0.00000369350062775",
            "extra": "mean: 63.7011711357 usec\nrounds: 13240"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-eager]",
            "value": 46274.568586,
            "unit": "iter/sec",
            "range": "stddev: 0.0000109049666936",
            "extra": "mean: 21.6101420404 usec\nrounds: 36035"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-eager]",
            "value": 31819.3450312,
            "unit": "iter/sec",
            "range": "stddev: 0.00000226364189623",
            "extra": "mean: 31.4274225011 usec\nrounds: 22222"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-compile]",
            "value": 6774.98449912,
            "unit": "iter/sec",
            "range": "stddev: 0.0000146959234888",
            "extra": "mean: 147.601813721 usec\nrounds: 3908"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-eager]",
            "value": 2585.27885166,
            "unit": "iter/sec",
            "range": "stddev: 0.0000127483840668",
            "extra": "mean: 386.805469498 usec\nrounds: 2590"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-compile]",
            "value": 6308.49711161,
            "unit": "iter/sec",
            "range": "stddev: 0.00000845711048377",
            "extra": "mean: 158.516360126 usec\nrounds: 2438"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[eager]",
            "value": 5708.12025945,
            "unit": "iter/sec",
            "range": "stddev: 0.0000181949218323",
            "extra": "mean: 175.189020999 usec\nrounds: 5912"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[compile]",
            "value": 5584.94792531,
            "unit": "iter/sec",
            "range": "stddev: 0.0000112670676805",
            "extra": "mean: 179.052699036 usec\nrounds: 3489"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[eager]",
            "value": 5380.87063931,
            "unit": "iter/sec",
            "range": "stddev: 0.0000425420341338",
            "extra": "mean: 185.8435311 usec\nrounds: 5131"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[compile]",
            "value": 5356.22499209,
            "unit": "iter/sec",
            "range": "stddev: 0.00000909330377331",
            "extra": "mean: 186.698654645 usec\nrounds: 3159"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-eager]",
            "value": 110819.219664,
            "unit": "iter/sec",
            "range": "stddev: 7.84897990344e-7",
            "extra": "mean: 9.02370548205 usec\nrounds: 82434"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-compile]",
            "value": 40016.2535925,
            "unit": "iter/sec",
            "range": "stddev: 0.0000076653498921",
            "extra": "mean: 24.9898456308 usec\nrounds: 27502"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-eager]",
            "value": 65744.5160393,
            "unit": "iter/sec",
            "range": "stddev: 0.00000145561498733",
            "extra": "mean: 15.2103941172 usec\nrounds: 53448"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-compile]",
            "value": 33151.2710757,
            "unit": "iter/sec",
            "range": "stddev: 0.00000764071361817",
            "extra": "mean: 30.1647559069 usec\nrounds: 21160"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-eager]",
            "value": 36811.0823569,
            "unit": "iter/sec",
            "range": "stddev: 0.00000185659673299",
            "extra": "mean: 27.165732056 usec\nrounds: 32332"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-compile]",
            "value": 19976.6380657,
            "unit": "iter/sec",
            "range": "stddev: 0.0000045122608262",
            "extra": "mean: 50.0584731379 usec\nrounds: 18064"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[eager]",
            "value": 35734.0609533,
            "unit": "iter/sec",
            "range": "stddev: 0.00000681401384846",
            "extra": "mean: 27.9845047924 usec\nrounds: 29569"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[compile]",
            "value": 7713.20360213,
            "unit": "iter/sec",
            "range": "stddev: 0.0000155043415428",
            "extra": "mean: 129.647815821 usec\nrounds: 6966"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[eager]",
            "value": 29850.0009776,
            "unit": "iter/sec",
            "range": "stddev: 0.00000377417294806",
            "extra": "mean: 33.5008364238 usec\nrounds: 21872"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile]",
            "value": 8377.19237987,
            "unit": "iter/sec",
            "range": "stddev: 0.0000121343638959",
            "extra": "mean: 119.371736335 usec\nrounds: 5480"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile-overhead]",
            "value": 4054.35662771,
            "unit": "iter/sec",
            "range": "stddev: 0.000147276794112",
            "extra": "mean: 246.648258115 usec\nrounds: 310"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[eager]",
            "value": 3648.03677762,
            "unit": "iter/sec",
            "range": "stddev: 0.0000148194372894",
            "extra": "mean: 274.120043454 usec\nrounds: 3518"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile]",
            "value": 3102.88861355,
            "unit": "iter/sec",
            "range": "stddev: 0.0000211570683766",
            "extra": "mean: 322.280340852 usec\nrounds: 2171"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile-overhead]",
            "value": 237.957195567,
            "unit": "iter/sec",
            "range": "stddev: 0.00197213473933",
            "extra": "mean: 4.20243648282 msec\nrounds: 1021"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[False]",
            "value": 6103.17558343,
            "unit": "iter/sec",
            "range": "stddev: 0.00000838838339302",
            "extra": "mean: 163.84912843 usec\nrounds: 5201"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[True]",
            "value": 6121.37048608,
            "unit": "iter/sec",
            "range": "stddev: 0.0000103587436156",
            "extra": "mean: 163.362110213 usec\nrounds: 5241"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[eager]",
            "value": 665.504979073,
            "unit": "iter/sec",
            "range": "stddev: 0.000044433833222",
            "extra": "mean: 1.50261835966 msec\nrounds: 645"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile]",
            "value": 721.210260068,
            "unit": "iter/sec",
            "range": "stddev: 0.0000883700519513",
            "extra": "mean: 1.38655819997 msec\nrounds: 610"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile-overhead]",
            "value": 1004.48282432,
            "unit": "iter/sec",
            "range": "stddev: 0.000137000437207",
            "extra": "mean: 995.537181706 usec\nrounds: 77"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[eager]",
            "value": 9823.29111457,
            "unit": "iter/sec",
            "range": "stddev: 0.00000649840976751",
            "extra": "mean: 101.798876602 usec\nrounds: 8452"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile]",
            "value": 7320.67952167,
            "unit": "iter/sec",
            "range": "stddev: 0.0000158206530721",
            "extra": "mean: 136.599341228 usec\nrounds: 6961"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile-overhead]",
            "value": 4952.52164647,
            "unit": "iter/sec",
            "range": "stddev: 0.00000813455307403",
            "extra": "mean: 201.917340576 usec\nrounds: 3309"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[eager]",
            "value": 2487.00898876,
            "unit": "iter/sec",
            "range": "stddev: 0.0000144095109554",
            "extra": "mean: 402.089419266 usec\nrounds: 2366"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile]",
            "value": 2860.5198164,
            "unit": "iter/sec",
            "range": "stddev: 0.0000207323367666",
            "extra": "mean: 349.586810854 usec\nrounds: 2522"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile-overhead]",
            "value": 3298.71924973,
            "unit": "iter/sec",
            "range": "stddev: 0.0000118897119547",
            "extra": "mean: 303.147956615 usec\nrounds: 2325"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-eager]",
            "value": 1161.90393059,
            "unit": "iter/sec",
            "range": "stddev: 0.0000331248735292",
            "extra": "mean: 860.656353483 usec\nrounds: 1177"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile]",
            "value": 1341.85040365,
            "unit": "iter/sec",
            "range": "stddev: 0.0000134396585863",
            "extra": "mean: 745.239556719 usec\nrounds: 1234"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile-overhead]",
            "value": 2117.79663093,
            "unit": "iter/sec",
            "range": "stddev: 0.0000321417923987",
            "extra": "mean: 472.188870921 usec\nrounds: 1046"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-eager]",
            "value": 780.935667044,
            "unit": "iter/sec",
            "range": "stddev: 0.0000237685207644",
            "extra": "mean: 1.28051521041 msec\nrounds: 770"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile]",
            "value": 1305.48943664,
            "unit": "iter/sec",
            "range": "stddev: 0.0000203818821476",
            "extra": "mean: 765.996240135 usec\nrounds: 1287"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile-overhead]",
            "value": 1995.94775191,
            "unit": "iter/sec",
            "range": "stddev: 0.00001355020794",
            "extra": "mean: 501.015118778 usec\nrounds: 943"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-eager]",
            "value": 1224.73857588,
            "unit": "iter/sec",
            "range": "stddev: 0.0000264369338529",
            "extra": "mean: 816.50077796 usec\nrounds: 1211"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile]",
            "value": 1429.71893668,
            "unit": "iter/sec",
            "range": "stddev: 0.0000264376773109",
            "extra": "mean: 699.438172318 usec\nrounds: 1329"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile-overhead]",
            "value": 2181.82559911,
            "unit": "iter/sec",
            "range": "stddev: 0.0000205041322538",
            "extra": "mean: 458.331775193 usec\nrounds: 845"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-eager]",
            "value": 749.408995817,
            "unit": "iter/sec",
            "range": "stddev: 0.0000367004314756",
            "extra": "mean: 1.33438483603 msec\nrounds: 738"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile]",
            "value": 1194.20635542,
            "unit": "iter/sec",
            "range": "stddev: 0.0000330204052969",
            "extra": "mean: 837.376216819 usec\nrounds: 1162"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile-overhead]",
            "value": 1325.30625847,
            "unit": "iter/sec",
            "range": "stddev: 0.000111703015383",
            "extra": "mean: 754.542577314 usec\nrounds: 880"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[eager]",
            "value": 340.921094571,
            "unit": "iter/sec",
            "range": "stddev: 0.000115753279397",
            "extra": "mean: 2.93323005213 msec\nrounds: 346"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile]",
            "value": 1286.99165551,
            "unit": "iter/sec",
            "range": "stddev: 0.0000305044661476",
            "extra": "mean: 777.005814853 usec\nrounds: 983"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile-overhead]",
            "value": 1853.20160083,
            "unit": "iter/sec",
            "range": "stddev: 0.0002376529765",
            "extra": "mean: 539.606699861 usec\nrounds: 130"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1056.70222468,
            "unit": "iter/sec",
            "range": "stddev: 0.0000497204601642",
            "extra": "mean: 946.340394338 usec\nrounds: 961"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1060.03578435,
            "unit": "iter/sec",
            "range": "stddev: 0.0000405239043215",
            "extra": "mean: 943.364379545 usec\nrounds: 972"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1149.91970321,
            "unit": "iter/sec",
            "range": "stddev: 0.0000403703488573",
            "extra": "mean: 869.62593754 usec\nrounds: 1153"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1149.68509496,
            "unit": "iter/sec",
            "range": "stddev: 0.0000570466597374",
            "extra": "mean: 869.803396067 usec\nrounds: 1154"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-True]",
            "value": 26.0524940749,
            "unit": "iter/sec",
            "range": "stddev: 0.0643060961007",
            "extra": "mean: 38.3840409722 msec\nrounds: 36"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-False]",
            "value": 37.4557187036,
            "unit": "iter/sec",
            "range": "stddev: 0.000654913334627",
            "extra": "mean: 26.6981928157 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-True]",
            "value": 37.2975569096,
            "unit": "iter/sec",
            "range": "stddev: 0.000834969757634",
            "extra": "mean: 26.8114075789 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-False]",
            "value": 37.7405397891,
            "unit": "iter/sec",
            "range": "stddev: 0.000692431215028",
            "extra": "mean: 26.4967063425 msec\nrounds: 38"
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
          "id": "b47e6cd73694bf1519bf38907aa5a9a9276c8381",
          "message": "[Test] Stop a store test from leaking a Redis connection that fails unrelated tests (#3066)",
          "timestamp": "2026-10-20T04:10:17-04:00",
          "tree_id": "254cc445ec14bd79b2e9e12a75cd832112f0d59d",
          "url": "https://github.com/pytorch/tensordict/commit/b47e6cd73694bf1519bf38907aa5a9a9276c8381"
        },
        "date": 1792484620941,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-2]",
            "value": 4559.5361478,
            "unit": "iter/sec",
            "range": "stddev: 0.00000650669565755",
            "extra": "mean: 219.320555334 usec\nrounds: 135"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-8]",
            "value": 3597.37756149,
            "unit": "iter/sec",
            "range": "stddev: 0.00000664797571421",
            "extra": "mean: 277.980273937 usec\nrounds: 3209"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-32]",
            "value": 2000.01204264,
            "unit": "iter/sec",
            "range": "stddev: 0.0000120162552927",
            "extra": "mean: 499.996989358 usec\nrounds: 1860"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-2]",
            "value": 4557.93525055,
            "unit": "iter/sec",
            "range": "stddev: 0.0000277890541001",
            "extra": "mean: 219.39758795 usec\nrounds: 4036"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-8]",
            "value": 3467.38661145,
            "unit": "iter/sec",
            "range": "stddev: 0.000024851179118",
            "extra": "mean: 288.401644252 usec\nrounds: 3261"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-32]",
            "value": 1791.25578982,
            "unit": "iter/sec",
            "range": "stddev: 0.000019084115986",
            "extra": "mean: 558.267560491 usec\nrounds: 1663"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-2]",
            "value": 2233.10584048,
            "unit": "iter/sec",
            "range": "stddev: 0.0000163427489021",
            "extra": "mean: 447.806808739 usec\nrounds: 73"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-8]",
            "value": 601.980887998,
            "unit": "iter/sec",
            "range": "stddev: 0.0000569678695493",
            "extra": "mean: 1.66118230651 msec\nrounds: 581"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-32]",
            "value": 150.290293457,
            "unit": "iter/sec",
            "range": "stddev: 0.000360596662265",
            "extra": "mean: 6.65378965599 msec\nrounds: 154"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-2]",
            "value": 2216.61452923,
            "unit": "iter/sec",
            "range": "stddev: 0.0000152434331058",
            "extra": "mean: 451.138430617 usec\nrounds: 2120"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-8]",
            "value": 590.777270786,
            "unit": "iter/sec",
            "range": "stddev: 0.0000437469870289",
            "extra": "mean: 1.6926852969 msec\nrounds: 603"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-32]",
            "value": 156.478904666,
            "unit": "iter/sec",
            "range": "stddev: 0.0000778765048563",
            "extra": "mean: 6.39063778043 msec\nrounds: 155"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-2]",
            "value": 3154.23356182,
            "unit": "iter/sec",
            "range": "stddev: 0.00000664677587138",
            "extra": "mean: 317.034227302 usec\nrounds: 2433"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-8]",
            "value": 846.466325924,
            "unit": "iter/sec",
            "range": "stddev: 0.0000116727826275",
            "extra": "mean: 1.18138190425 msec\nrounds: 824"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-32]",
            "value": 216.911228713,
            "unit": "iter/sec",
            "range": "stddev: 0.000035474670337",
            "extra": "mean: 4.61018088336 msec\nrounds: 214"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-2]",
            "value": 3118.59424816,
            "unit": "iter/sec",
            "range": "stddev: 0.0000078689174488",
            "extra": "mean: 320.657296341 usec\nrounds: 2932"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-8]",
            "value": 833.292809386,
            "unit": "iter/sec",
            "range": "stddev: 0.0000381524287855",
            "extra": "mean: 1.20005835732 msec\nrounds: 817"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-32]",
            "value": 213.03888811,
            "unit": "iter/sec",
            "range": "stddev: 0.000141245148297",
            "extra": "mean: 4.69397868564 msec\nrounds: 213"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-2]",
            "value": 3323.19538462,
            "unit": "iter/sec",
            "range": "stddev: 0.00000947179480541",
            "extra": "mean: 300.915198856 usec\nrounds: 3013"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-8]",
            "value": 913.616003959,
            "unit": "iter/sec",
            "range": "stddev: 0.0000147974138376",
            "extra": "mean: 1.09455175442 msec\nrounds: 896"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-32]",
            "value": 232.647546628,
            "unit": "iter/sec",
            "range": "stddev: 0.0000364131994926",
            "extra": "mean: 4.29834749815 msec\nrounds: 231"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-2]",
            "value": 3580.21618386,
            "unit": "iter/sec",
            "range": "stddev: 0.0000100516218252",
            "extra": "mean: 279.312742205 usec\nrounds: 3305"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-8]",
            "value": 940.242861962,
            "unit": "iter/sec",
            "range": "stddev: 0.0000409301296155",
            "extra": "mean: 1.06355500313 msec\nrounds: 945"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-32]",
            "value": 241.908258713,
            "unit": "iter/sec",
            "range": "stddev: 0.000135688014681",
            "extra": "mean: 4.13379851239 msec\nrounds: 246"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-128]",
            "value": 191.461541384,
            "unit": "iter/sec",
            "range": "stddev: 0.0000338308175598",
            "extra": "mean: 5.22298103719 msec\nrounds: 188"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-256]",
            "value": 98.7102080079,
            "unit": "iter/sec",
            "range": "stddev: 0.00030297441069",
            "extra": "mean: 10.1306644995 msec\nrounds: 96"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-128]",
            "value": 183.138689818,
            "unit": "iter/sec",
            "range": "stddev: 0.000138544162748",
            "extra": "mean: 5.46034265612 msec\nrounds: 180"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-256]",
            "value": 94.8507038813,
            "unit": "iter/sec",
            "range": "stddev: 0.000137965177737",
            "extra": "mean: 10.5428843338 msec\nrounds: 87"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-128]",
            "value": 12.7655695519,
            "unit": "iter/sec",
            "range": "stddev: 0.00170482182725",
            "extra": "mean: 78.3357135718 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-256]",
            "value": 6.27273447873,
            "unit": "iter/sec",
            "range": "stddev: 0.000423904426416",
            "extra": "mean: 159.420106716 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-128]",
            "value": 12.7140318605,
            "unit": "iter/sec",
            "range": "stddev: 0.00154415328405",
            "extra": "mean: 78.6532557863 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-256]",
            "value": 6.22003706513,
            "unit": "iter/sec",
            "range": "stddev: 0.000683621822232",
            "extra": "mean: 160.770746143 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-128]",
            "value": 23.8951911518,
            "unit": "iter/sec",
            "range": "stddev: 0.000120416689685",
            "extra": "mean: 41.8494245829 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-256]",
            "value": 11.9006605217,
            "unit": "iter/sec",
            "range": "stddev: 0.000478372021574",
            "extra": "mean: 84.0289493323 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-128]",
            "value": 23.0154901971,
            "unit": "iter/sec",
            "range": "stddev: 0.0031626670768",
            "extra": "mean: 43.4489985413 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-256]",
            "value": 11.836746515,
            "unit": "iter/sec",
            "range": "stddev: 0.000385902050069",
            "extra": "mean: 84.4826742493 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-128]",
            "value": 26.509936062,
            "unit": "iter/sec",
            "range": "stddev: 0.00105386759338",
            "extra": "mean: 37.7217054639 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-256]",
            "value": 13.5296568624,
            "unit": "iter/sec",
            "range": "stddev: 0.00227413461224",
            "extra": "mean: 73.9117044998 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-128]",
            "value": 25.7515110437,
            "unit": "iter/sec",
            "range": "stddev: 0.000216140402112",
            "extra": "mean: 38.8326727043 msec\nrounds: 27"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-256]",
            "value": 13.0916590383,
            "unit": "iter/sec",
            "range": "stddev: 0.00205255481035",
            "extra": "mean: 76.3845129997 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_mask_many_leaves[cuda]",
            "value": 6558.10181741,
            "unit": "iter/sec",
            "range": "stddev: 0.0000154411786845",
            "extra": "mean: 152.483146472 usec\nrounds: 41"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-None]",
            "value": 6567.85927912,
            "unit": "iter/sec",
            "range": "stddev: 0.00000517074103925",
            "extra": "mean: 152.256611706 usec\nrounds: 1403"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-cpu]",
            "value": 6650.43137007,
            "unit": "iter/sec",
            "range": "stddev: 0.0000056545951689",
            "extra": "mean: 150.366185944 usec\nrounds: 4920"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to",
            "value": 22065.1555179,
            "unit": "iter/sec",
            "range": "stddev: 0.00000266875531598",
            "extra": "mean: 45.3203241278 usec\nrounds: 10610"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_nonblocking",
            "value": 21957.57531,
            "unit": "iter/sec",
            "range": "stddev: 0.00000339000072878",
            "extra": "mean: 45.5423691314 usec\nrounds: 11189"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[add]",
            "value": 10625.4306529,
            "unit": "iter/sec",
            "range": "stddev: 0.00000782580975981",
            "extra": "mean: 94.1138324333 usec\nrounds: 179"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[sub]",
            "value": 10560.9212099,
            "unit": "iter/sec",
            "range": "stddev: 0.00000507296939145",
            "extra": "mean: 94.688709453 usec\nrounds: 8123"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_blocking",
            "value": 55.84236488,
            "unit": "iter/sec",
            "range": "stddev: 0.000982356290503",
            "extra": "mean: 17.9075510528 msec\nrounds: 57"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_global_sync",
            "value": 96.0885544232,
            "unit": "iter/sec",
            "range": "stddev: 0.0119468272925",
            "extra": "mean: 10.4070667521 msec\nrounds: 109"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_event_sync",
            "value": 98.0856858281,
            "unit": "iter/sec",
            "range": "stddev: 0.0000989021118338",
            "extra": "mean: 10.195167537 msec\nrounds: 95"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_default",
            "value": 95.964600944,
            "unit": "iter/sec",
            "range": "stddev: 0.000100078567233",
            "extra": "mean: 10.4205091269 msec\nrounds: 95"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_cpu_to_cpu_default",
            "value": 3856.65947807,
            "unit": "iter/sec",
            "range": "stddev: 0.0000047464521046",
            "extra": "mean: 259.291753831 usec\nrounds: 3859"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[False-None]",
            "value": 295.204715618,
            "unit": "iter/sec",
            "range": "stddev: 0.0000711617371422",
            "extra": "mean: 3.38747976266 msec\nrounds: 295"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[default-None]",
            "value": 245.300653864,
            "unit": "iter/sec",
            "range": "stddev: 0.0382673811717",
            "extra": "mean: 4.07662998141 msec\nrounds: 427"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[reduce-overhead-None]",
            "value": 449.881573869,
            "unit": "iter/sec",
            "range": "stddev: 0.000263554592461",
            "extra": "mean: 2.22280719657 msec\nrounds: 428"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[False-False-None]",
            "value": 488.950326769,
            "unit": "iter/sec",
            "range": "stddev: 0.0000316208815482",
            "extra": "mean: 2.04519752877 msec\nrounds: 465"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-False-None]",
            "value": 637.182589628,
            "unit": "iter/sec",
            "range": "stddev: 0.0000585082344167",
            "extra": "mean: 1.56940885749 msec\nrounds: 597"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[within-False-None]",
            "value": 194.989008783,
            "unit": "iter/sec",
            "range": "stddev: 0.0000446606868726",
            "extra": "mean: 5.12849419689 msec\nrounds: 198"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-default-None]",
            "value": 115.644273228,
            "unit": "iter/sec",
            "range": "stddev: 0.000125890904206",
            "extra": "mean: 8.64720726834 msec\nrounds: 108"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[False-False-None]",
            "value": 140.623054068,
            "unit": "iter/sec",
            "range": "stddev: 0.000101112335789",
            "extra": "mean: 7.11120951418 msec\nrounds: 142"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[True-False-None]",
            "value": 264.407186558,
            "unit": "iter/sec",
            "range": "stddev: 0.0000559754127721",
            "extra": "mean: 3.78204546185 msec\nrounds: 262"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[within-False-None]",
            "value": 108.893465546,
            "unit": "iter/sec",
            "range": "stddev: 0.000706415054499",
            "extra": "mean: 9.18328749103 msec\nrounds: 110"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 11.6113205023,
            "unit": "iter/sec",
            "range": "stddev: 0.000755237897588",
            "extra": "mean: 86.1228488007 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_archive",
            "value": 4.58076948395,
            "unit": "iter/sec",
            "range": "stddev: 0.000904783388486",
            "extra": "mean: 218.303934198 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_zarr",
            "value": 0.466580987859,
            "unit": "iter/sec",
            "range": "stddev: 0.12192469525",
            "extra": "mean: 2.1432506382 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 42.6295717104,
            "unit": "iter/sec",
            "range": "stddev: 0.000299536687812",
            "extra": "mean: 23.4578945994 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.40628850209,
            "unit": "iter/sec",
            "range": "stddev: 0.0463410239272",
            "extra": "mean: 106.311857198 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 10.9193842435,
            "unit": "iter/sec",
            "range": "stddev: 0.000585044908406",
            "extra": "mean: 91.5802555988 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 43.8174740405,
            "unit": "iter/sec",
            "range": "stddev: 0.0000985928990147",
            "extra": "mean: 22.8219453973 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-compile]",
            "value": 4844.13187038,
            "unit": "iter/sec",
            "range": "stddev: 0.00000724925061923",
            "extra": "mean: 206.435338004 usec\nrounds: 1713"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-eager]",
            "value": 8204.78630535,
            "unit": "iter/sec",
            "range": "stddev: 0.00000588004852282",
            "extra": "mean: 121.880078625 usec\nrounds: 6955"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-compile]",
            "value": 6486.7892237,
            "unit": "iter/sec",
            "range": "stddev: 0.00000670340169401",
            "extra": "mean: 154.159471738 usec\nrounds: 2777"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-eager]",
            "value": 5761.24849742,
            "unit": "iter/sec",
            "range": "stddev: 0.000015835071135",
            "extra": "mean: 173.573488532 usec\nrounds: 5451"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-compile]",
            "value": 21588.891087,
            "unit": "iter/sec",
            "range": "stddev: 0.00000258097357103",
            "extra": "mean: 46.3201188042 usec\nrounds: 16119"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-eager]",
            "value": 30442.3508822,
            "unit": "iter/sec",
            "range": "stddev: 0.00000134968393342",
            "extra": "mean: 32.8489742421 usec\nrounds: 25746"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-compile]",
            "value": 50136.0228267,
            "unit": "iter/sec",
            "range": "stddev: 0.00000336299786705",
            "extra": "mean: 19.9457384854 usec\nrounds: 37452"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-eager]",
            "value": 16086.3175128,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113128777944",
            "extra": "mean: 62.1646314767 usec\nrounds: 14793"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-compile]",
            "value": 3761.32778316,
            "unit": "iter/sec",
            "range": "stddev: 0.0000088671465418",
            "extra": "mean: 265.863561394 usec\nrounds: 1074"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-eager]",
            "value": 5192.60456265,
            "unit": "iter/sec",
            "range": "stddev: 0.00000723042289203",
            "extra": "mean: 192.581581735 usec\nrounds: 4899"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-compile]",
            "value": 5712.12423362,
            "unit": "iter/sec",
            "range": "stddev: 0.00000946504204445",
            "extra": "mean: 175.066220394 usec\nrounds: 3321"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-eager]",
            "value": 16021.1447178,
            "unit": "iter/sec",
            "range": "stddev: 0.00000453520236353",
            "extra": "mean: 62.4175124575 usec\nrounds: 14113"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-compile]",
            "value": 4189.606204,
            "unit": "iter/sec",
            "range": "stddev: 0.00000590731570233",
            "extra": "mean: 238.685917317 usec\nrounds: 1395"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-eager]",
            "value": 1924.84548118,
            "unit": "iter/sec",
            "range": "stddev: 0.0000167930957307",
            "extra": "mean: 519.522221279 usec\nrounds: 1912"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-eager]",
            "value": 4320.89671483,
            "unit": "iter/sec",
            "range": "stddev: 0.00000537921917877",
            "extra": "mean: 231.433442176 usec\nrounds: 4118"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-compile]",
            "value": 3767.25822178,
            "unit": "iter/sec",
            "range": "stddev: 0.00000787933444631",
            "extra": "mean: 265.445037512 usec\nrounds: 1356"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-eager]",
            "value": 13091.0839351,
            "unit": "iter/sec",
            "range": "stddev: 0.00000333640480898",
            "extra": "mean: 76.3878686408 usec\nrounds: 11480"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-compile]",
            "value": 6000.54337314,
            "unit": "iter/sec",
            "range": "stddev: 0.00000604764194774",
            "extra": "mean: 166.651574335 usec\nrounds: 3355"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-eager]",
            "value": 2223.22332824,
            "unit": "iter/sec",
            "range": "stddev: 0.0000158241274178",
            "extra": "mean: 449.797367316 usec\nrounds: 2176"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-compile]",
            "value": 4187.64365109,
            "unit": "iter/sec",
            "range": "stddev: 0.00000676188590404",
            "extra": "mean: 238.797778254 usec\nrounds: 1406"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-compile]",
            "value": 34407.7973929,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022204844585",
            "extra": "mean: 29.0631797375 usec\nrounds: 22237"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-eager]",
            "value": 54505.6349011,
            "unit": "iter/sec",
            "range": "stddev: 0.00000120503906264",
            "extra": "mean: 18.3467269359 usec\nrounds: 46969"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-compile]",
            "value": 36077.8789686,
            "unit": "iter/sec",
            "range": "stddev: 0.00000212141962249",
            "extra": "mean: 27.7178156973 usec\nrounds: 26596"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-eager]",
            "value": 19909.3184002,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113143746942",
            "extra": "mean: 50.2277365753 usec\nrounds: 17158"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-compile]",
            "value": 3992.00371665,
            "unit": "iter/sec",
            "range": "stddev: 0.000215394702985",
            "extra": "mean: 250.500768782 usec\nrounds: 95"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-eager]",
            "value": 310.334927075,
            "unit": "iter/sec",
            "range": "stddev: 0.0000595017730191",
            "extra": "mean: 3.2223250197 msec\nrounds: 304"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-compile]",
            "value": 5189.74396197,
            "unit": "iter/sec",
            "range": "stddev: 0.000205806478076",
            "extra": "mean: 192.687733215 usec\nrounds: 90"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-eager]",
            "value": 350.164991945,
            "unit": "iter/sec",
            "range": "stddev: 0.0000302105882569",
            "extra": "mean: 2.85579661875 msec\nrounds: 341"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-compile]",
            "value": 6130.37411493,
            "unit": "iter/sec",
            "range": "stddev: 0.00000629466110874",
            "extra": "mean: 163.122181657 usec\nrounds: 3968"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-eager]",
            "value": 18531.6274982,
            "unit": "iter/sec",
            "range": "stddev: 0.0000124157607308",
            "extra": "mean: 53.9618012557 usec\nrounds: 14582"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-compile]",
            "value": 7891.08695128,
            "unit": "iter/sec",
            "range": "stddev: 0.00000462038362296",
            "extra": "mean: 126.725254223 usec\nrounds: 4559"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-eager]",
            "value": 21791.8733075,
            "unit": "iter/sec",
            "range": "stddev: 0.0000117792692217",
            "extra": "mean: 45.8886661963 usec\nrounds: 17983"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-compile]",
            "value": 7806.88567267,
            "unit": "iter/sec",
            "range": "stddev: 0.00000384689787633",
            "extra": "mean: 128.09205129 usec\nrounds: 7038"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-eager]",
            "value": 21767.7921152,
            "unit": "iter/sec",
            "range": "stddev: 0.0000115330113758",
            "extra": "mean: 45.9394317396 usec\nrounds: 18522"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-compile]",
            "value": 12126.2922665,
            "unit": "iter/sec",
            "range": "stddev: 0.00000649295490269",
            "extra": "mean: 82.4654377465 usec\nrounds: 9777"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-eager]",
            "value": 46163.8559973,
            "unit": "iter/sec",
            "range": "stddev: 0.0000106014296527",
            "extra": "mean: 21.6619686202 usec\nrounds: 34494"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-compile]",
            "value": 15859.7789436,
            "unit": "iter/sec",
            "range": "stddev: 0.00000358230943157",
            "extra": "mean: 63.0525812216 usec\nrounds: 2388"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-eager]",
            "value": 46963.1886293,
            "unit": "iter/sec",
            "range": "stddev: 0.0000108691488127",
            "extra": "mean: 21.2932730759 usec\nrounds: 37356"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-compile]",
            "value": 15788.8170869,
            "unit": "iter/sec",
            "range": "stddev: 0.00000331385017312",
            "extra": "mean: 63.3359671276 usec\nrounds: 13543"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-eager]",
            "value": 46899.6208622,
            "unit": "iter/sec",
            "range": "stddev: 0.0000109581799015",
            "extra": "mean: 21.3221339878 usec\nrounds: 35907"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-compile]",
            "value": 10420.6337391,
            "unit": "iter/sec",
            "range": "stddev: 0.0000134656644639",
            "extra": "mean: 95.9634533792 usec\nrounds: 10477"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-eager]",
            "value": 46593.5683816,
            "unit": "iter/sec",
            "range": "stddev: 0.000010367730486",
            "extra": "mean: 21.4621896269 usec\nrounds: 36257"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-compile]",
            "value": 16003.21367,
            "unit": "iter/sec",
            "range": "stddev: 0.00000327524967351",
            "extra": "mean: 62.4874491225 usec\nrounds: 13751"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-eager]",
            "value": 43790.7098962,
            "unit": "iter/sec",
            "range": "stddev: 0.0000111045617146",
            "extra": "mean: 22.8358937859 usec\nrounds: 38401"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-compile]",
            "value": 15519.9771151,
            "unit": "iter/sec",
            "range": "stddev: 0.00000321589473617",
            "extra": "mean: 64.4330847002 usec\nrounds: 13505"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-eager]",
            "value": 45854.6365352,
            "unit": "iter/sec",
            "range": "stddev: 0.0000109327161931",
            "extra": "mean: 21.8080455012 usec\nrounds: 39230"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-eager]",
            "value": 34107.327266,
            "unit": "iter/sec",
            "range": "stddev: 0.00000345167167686",
            "extra": "mean: 29.3192132061 usec\nrounds: 23507"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-compile]",
            "value": 7108.65470969,
            "unit": "iter/sec",
            "range": "stddev: 0.00000538722132645",
            "extra": "mean: 140.673593083 usec\nrounds: 4333"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-eager]",
            "value": 2731.60614899,
            "unit": "iter/sec",
            "range": "stddev: 0.0000120979914713",
            "extra": "mean: 366.084986435 usec\nrounds: 2561"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-compile]",
            "value": 6402.67450763,
            "unit": "iter/sec",
            "range": "stddev: 0.00000562817153862",
            "extra": "mean: 156.184731679 usec\nrounds: 3532"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[eager]",
            "value": 6155.51605931,
            "unit": "iter/sec",
            "range": "stddev: 0.00000402896673975",
            "extra": "mean: 162.455916021 usec\nrounds: 5881"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[compile]",
            "value": 5682.94798892,
            "unit": "iter/sec",
            "range": "stddev: 0.00000522150309004",
            "extra": "mean: 175.965010053 usec\nrounds: 3791"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[eager]",
            "value": 5844.31127981,
            "unit": "iter/sec",
            "range": "stddev: 0.00000466268712075",
            "extra": "mean: 171.10656023 usec\nrounds: 5437"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[compile]",
            "value": 5349.30353826,
            "unit": "iter/sec",
            "range": "stddev: 0.00000787650768551",
            "extra": "mean: 186.940223685 usec\nrounds: 3648"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-eager]",
            "value": 109585.182705,
            "unit": "iter/sec",
            "range": "stddev: 6.05657284677e-7",
            "extra": "mean: 9.12532128261 usec\nrounds: 83264"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-compile]",
            "value": 39473.1500838,
            "unit": "iter/sec",
            "range": "stddev: 0.00000918635754704",
            "extra": "mean: 25.3336761286 usec\nrounds: 25582"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-eager]",
            "value": 64886.942426,
            "unit": "iter/sec",
            "range": "stddev: 0.00000167265001956",
            "extra": "mean: 15.4114211984 usec\nrounds: 54198"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-compile]",
            "value": 33821.2779402,
            "unit": "iter/sec",
            "range": "stddev: 0.00000252184586832",
            "extra": "mean: 29.5671855383 usec\nrounds: 23737"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-eager]",
            "value": 36722.6938386,
            "unit": "iter/sec",
            "range": "stddev: 0.00000193249473801",
            "extra": "mean: 27.2311177496 usec\nrounds: 33955"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-compile]",
            "value": 25258.2326936,
            "unit": "iter/sec",
            "range": "stddev: 0.00000303409509553",
            "extra": "mean: 39.5910518417 usec\nrounds: 18278"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[eager]",
            "value": 37444.8133365,
            "unit": "iter/sec",
            "range": "stddev: 0.00000592906612138",
            "extra": "mean: 26.7059683544 usec\nrounds: 25707"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[compile]",
            "value": 7741.13016421,
            "unit": "iter/sec",
            "range": "stddev: 0.0000507052942257",
            "extra": "mean: 129.18010404 usec\nrounds: 7361"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[eager]",
            "value": 32237.5920735,
            "unit": "iter/sec",
            "range": "stddev: 0.00000299583977637",
            "extra": "mean: 31.0196865115 usec\nrounds: 22401"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile]",
            "value": 7817.06656646,
            "unit": "iter/sec",
            "range": "stddev: 0.0000696880560903",
            "extra": "mean: 127.925225082 usec\nrounds: 5723"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile-overhead]",
            "value": 5105.40966321,
            "unit": "iter/sec",
            "range": "stddev: 0.000175172242343",
            "extra": "mean: 195.870667776 usec\nrounds: 316"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[eager]",
            "value": 3663.82671163,
            "unit": "iter/sec",
            "range": "stddev: 0.0000965414401821",
            "extra": "mean: 272.938672789 usec\nrounds: 3503"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile]",
            "value": 3076.01240473,
            "unit": "iter/sec",
            "range": "stddev: 0.00000939045969839",
            "extra": "mean: 325.096218228 usec\nrounds: 2543"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile-overhead]",
            "value": 245.630883943,
            "unit": "iter/sec",
            "range": "stddev: 0.00211232389394",
            "extra": "mean: 4.07114929503 msec\nrounds: 1027"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[False]",
            "value": 6291.62224922,
            "unit": "iter/sec",
            "range": "stddev: 0.0000106917609704",
            "extra": "mean: 158.941519435 usec\nrounds: 5182"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[True]",
            "value": 6195.22774791,
            "unit": "iter/sec",
            "range": "stddev: 0.00000717374270162",
            "extra": "mean: 161.414566291 usec\nrounds: 5446"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[eager]",
            "value": 608.445425535,
            "unit": "iter/sec",
            "range": "stddev: 0.000357798071757",
            "extra": "mean: 1.64353277719 msec\nrounds: 615"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile]",
            "value": 691.217470056,
            "unit": "iter/sec",
            "range": "stddev: 0.0000775674443431",
            "extra": "mean: 1.44672269339 msec\nrounds: 626"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile-overhead]",
            "value": 1073.06053681,
            "unit": "iter/sec",
            "range": "stddev: 0.0000810127843403",
            "extra": "mean: 931.913872232 usec\nrounds: 86"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[eager]",
            "value": 9894.18699818,
            "unit": "iter/sec",
            "range": "stddev: 0.00000613602423102",
            "extra": "mean: 101.069446149 usec\nrounds: 8511"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile]",
            "value": 7677.37110372,
            "unit": "iter/sec",
            "range": "stddev: 0.0000448000818601",
            "extra": "mean: 130.252919455 usec\nrounds: 6954"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile-overhead]",
            "value": 4779.55985962,
            "unit": "iter/sec",
            "range": "stddev: 0.000007985881956",
            "extra": "mean: 209.224286204 usec\nrounds: 3326"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[eager]",
            "value": 2383.29769174,
            "unit": "iter/sec",
            "range": "stddev: 0.0000171760355806",
            "extra": "mean: 419.586694296 usec\nrounds: 2447"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile]",
            "value": 2858.80019515,
            "unit": "iter/sec",
            "range": "stddev: 0.000010502812165",
            "extra": "mean: 349.797093794 usec\nrounds: 2760"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile-overhead]",
            "value": 3279.25624491,
            "unit": "iter/sec",
            "range": "stddev: 0.0000103006660701",
            "extra": "mean: 304.947196961 usec\nrounds: 1808"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-eager]",
            "value": 1186.95555353,
            "unit": "iter/sec",
            "range": "stddev: 0.0000431029664287",
            "extra": "mean: 842.491529719 usec\nrounds: 1244"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile]",
            "value": 1148.29177893,
            "unit": "iter/sec",
            "range": "stddev: 0.00054163390115",
            "extra": "mean: 870.858799439 usec\nrounds: 299"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile-overhead]",
            "value": 2141.62323648,
            "unit": "iter/sec",
            "range": "stddev: 0.0000172214594976",
            "extra": "mean: 466.935538879 usec\nrounds: 1351"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-eager]",
            "value": 753.066316228,
            "unit": "iter/sec",
            "range": "stddev: 0.00027253978097",
            "extra": "mean: 1.32790430066 msec\nrounds: 755"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile]",
            "value": 1247.94352175,
            "unit": "iter/sec",
            "range": "stddev: 0.00027396905036",
            "extra": "mean: 801.318314946 usec\nrounds: 1321"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile-overhead]",
            "value": 2043.70363915,
            "unit": "iter/sec",
            "range": "stddev: 0.0000178141453244",
            "extra": "mean: 489.307735643 usec\nrounds: 1233"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-eager]",
            "value": 1241.78925302,
            "unit": "iter/sec",
            "range": "stddev: 0.0000174297862548",
            "extra": "mean: 805.289623478 usec\nrounds: 1187"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile]",
            "value": 1434.45179382,
            "unit": "iter/sec",
            "range": "stddev: 0.0000204429231682",
            "extra": "mean: 697.130432901 usec\nrounds: 1370"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile-overhead]",
            "value": 2219.88329013,
            "unit": "iter/sec",
            "range": "stddev: 0.0000140763563277",
            "extra": "mean: 450.474132782 usec\nrounds: 1091"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-eager]",
            "value": 754.865189181,
            "unit": "iter/sec",
            "range": "stddev: 0.0000193390586452",
            "extra": "mean: 1.32473985333 msec\nrounds: 744"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile]",
            "value": 1152.33791648,
            "unit": "iter/sec",
            "range": "stddev: 0.000334430856803",
            "extra": "mean: 867.801003249 usec\nrounds: 1197"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile-overhead]",
            "value": 1351.54579668,
            "unit": "iter/sec",
            "range": "stddev: 0.0000472653918776",
            "extra": "mean: 739.893537056 usec\nrounds: 1037"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[eager]",
            "value": 348.560456595,
            "unit": "iter/sec",
            "range": "stddev: 0.0000380394705866",
            "extra": "mean: 2.86894276468 msec\nrounds: 344"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile]",
            "value": 1290.72556952,
            "unit": "iter/sec",
            "range": "stddev: 0.000213249807072",
            "extra": "mean: 774.758030378 usec\nrounds: 397"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile-overhead]",
            "value": 1930.66366069,
            "unit": "iter/sec",
            "range": "stddev: 0.0000444706457382",
            "extra": "mean: 517.956607543 usec\nrounds: 130"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1032.38762665,
            "unit": "iter/sec",
            "range": "stddev: 0.0000475359102506",
            "extra": "mean: 968.628424237 usec\nrounds: 976"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1056.9788844,
            "unit": "iter/sec",
            "range": "stddev: 0.0000446441692801",
            "extra": "mean: 946.092693774 usec\nrounds: 1035"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1159.25305048,
            "unit": "iter/sec",
            "range": "stddev: 0.0000339204021908",
            "extra": "mean: 862.624428363 usec\nrounds: 1146"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1154.47492474,
            "unit": "iter/sec",
            "range": "stddev: 0.0000376718131468",
            "extra": "mean: 866.19464708 usec\nrounds: 1147"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-True]",
            "value": 36.8310795795,
            "unit": "iter/sec",
            "range": "stddev: 0.000475638462212",
            "extra": "mean: 27.1509825782 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-False]",
            "value": 37.4097302197,
            "unit": "iter/sec",
            "range": "stddev: 0.00040664802585",
            "extra": "mean: 26.7310134055 msec\nrounds: 37"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-True]",
            "value": 37.9441905646,
            "unit": "iter/sec",
            "range": "stddev: 0.000457064396761",
            "extra": "mean: 26.3544955136 msec\nrounds: 39"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-False]",
            "value": 38.1824231708,
            "unit": "iter/sec",
            "range": "stddev: 0.000403252288492",
            "extra": "mean: 26.1900612103 msec\nrounds: 38"
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
          "id": "3dfc156d97db4fd6b401f059dea986be780178c5",
          "message": "[BugFix] Stop from_store from leaving a Redis connection open when reading the metadata fails (#3067)",
          "timestamp": "2026-10-20T04:10:32-04:00",
          "tree_id": "cc117c7c7be7a1c0dc57cd97135724abf5281968",
          "url": "https://github.com/pytorch/tensordict/commit/3dfc156d97db4fd6b401f059dea986be780178c5"
        },
        "date": 1792484668859,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-2]",
            "value": 4495.68311173,
            "unit": "iter/sec",
            "range": "stddev: 0.000007670325211",
            "extra": "mean: 222.435606592 usec\nrounds: 132"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-8]",
            "value": 3589.91626191,
            "unit": "iter/sec",
            "range": "stddev: 0.00000781557610743",
            "extra": "mean: 278.558029504 usec\nrounds: 3095"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-32]",
            "value": 2035.26168878,
            "unit": "iter/sec",
            "range": "stddev: 0.00000859103342076",
            "extra": "mean: 491.337308373 usec\nrounds: 1819"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-2]",
            "value": 4338.68312539,
            "unit": "iter/sec",
            "range": "stddev: 0.0000102887139223",
            "extra": "mean: 230.484681895 usec\nrounds: 4004"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-8]",
            "value": 3486.98148322,
            "unit": "iter/sec",
            "range": "stddev: 0.000012127402432",
            "extra": "mean: 286.780989464 usec\nrounds: 3211"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-32]",
            "value": 1804.38369707,
            "unit": "iter/sec",
            "range": "stddev: 0.00000904710389482",
            "extra": "mean: 554.205849688 usec\nrounds: 1664"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-2]",
            "value": 2273.94700692,
            "unit": "iter/sec",
            "range": "stddev: 0.00000812877745857",
            "extra": "mean: 439.763986126 usec\nrounds: 72"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-8]",
            "value": 610.416181662,
            "unit": "iter/sec",
            "range": "stddev: 0.0000317987560986",
            "extra": "mean: 1.63822655762 msec\nrounds: 599"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-32]",
            "value": 155.144530097,
            "unit": "iter/sec",
            "range": "stddev: 0.000065610228884",
            "extra": "mean: 6.44560268657 msec\nrounds: 153"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-2]",
            "value": 2272.38536731,
            "unit": "iter/sec",
            "range": "stddev: 0.0000124933306023",
            "extra": "mean: 440.066202848 usec\nrounds: 2128"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-8]",
            "value": 609.62908252,
            "unit": "iter/sec",
            "range": "stddev: 0.0000276539304453",
            "extra": "mean: 1.64034169083 msec\nrounds: 592"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-32]",
            "value": 154.629702136,
            "unit": "iter/sec",
            "range": "stddev: 0.0000670124863393",
            "extra": "mean: 6.46706283585 msec\nrounds: 152"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-2]",
            "value": 3096.49063024,
            "unit": "iter/sec",
            "range": "stddev: 0.00000723235056412",
            "extra": "mean: 322.946237988 usec\nrounds: 2281"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-8]",
            "value": 830.909807942,
            "unit": "iter/sec",
            "range": "stddev: 0.000011920878185",
            "extra": "mean: 1.20350005553 msec\nrounds: 795"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-32]",
            "value": 212.136981177,
            "unit": "iter/sec",
            "range": "stddev: 0.0000295487300746",
            "extra": "mean: 4.71393528112 msec\nrounds: 210"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-2]",
            "value": 3066.18921108,
            "unit": "iter/sec",
            "range": "stddev: 0.0000105990100943",
            "extra": "mean: 326.13773357 usec\nrounds: 2885"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-8]",
            "value": 821.726753135,
            "unit": "iter/sec",
            "range": "stddev: 0.0000398846773054",
            "extra": "mean: 1.21694954702 msec\nrounds: 810"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-32]",
            "value": 211.695903518,
            "unit": "iter/sec",
            "range": "stddev: 0.0000336489389719",
            "extra": "mean: 4.72375697112 msec\nrounds: 208"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-2]",
            "value": 3599.15054873,
            "unit": "iter/sec",
            "range": "stddev: 0.00000548001162749",
            "extra": "mean: 277.843337327 usec\nrounds: 2988"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-8]",
            "value": 962.480497769,
            "unit": "iter/sec",
            "range": "stddev: 0.0000152950984834",
            "extra": "mean: 1.03898209088 msec\nrounds: 934"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-32]",
            "value": 246.224847158,
            "unit": "iter/sec",
            "range": "stddev: 0.0000284308121825",
            "extra": "mean: 4.06132854398 msec\nrounds: 243"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-2]",
            "value": 3579.19840457,
            "unit": "iter/sec",
            "range": "stddev: 0.00000632047511197",
            "extra": "mean: 279.392167454 usec\nrounds: 3258"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-8]",
            "value": 959.739913975,
            "unit": "iter/sec",
            "range": "stddev: 0.0000147359543718",
            "extra": "mean: 1.04194895454 msec\nrounds: 929"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-32]",
            "value": 244.183354017,
            "unit": "iter/sec",
            "range": "stddev: 0.0000508664071482",
            "extra": "mean: 4.09528325149 msec\nrounds: 242"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-128]",
            "value": 190.171009155,
            "unit": "iter/sec",
            "range": "stddev: 0.0000610757204105",
            "extra": "mean: 5.25842505882 msec\nrounds: 187"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-256]",
            "value": 98.8740149066,
            "unit": "iter/sec",
            "range": "stddev: 0.000275245051696",
            "extra": "mean: 10.1138807901 msec\nrounds: 95"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-128]",
            "value": 183.20227966,
            "unit": "iter/sec",
            "range": "stddev: 0.0000577199683203",
            "extra": "mean: 5.45844736132 msec\nrounds: 180"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-256]",
            "value": 94.5650335284,
            "unit": "iter/sec",
            "range": "stddev: 0.00006373190518",
            "extra": "mean: 10.5747332041 msec\nrounds: 88"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-128]",
            "value": 13.1484800525,
            "unit": "iter/sec",
            "range": "stddev: 0.000601549097393",
            "extra": "mean: 76.0544181539 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-256]",
            "value": 6.10367150031,
            "unit": "iter/sec",
            "range": "stddev: 0.000514340701328",
            "extra": "mean: 163.835815861 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-128]",
            "value": 12.6309309672,
            "unit": "iter/sec",
            "range": "stddev: 0.0024765713382",
            "extra": "mean: 79.1707280005 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-256]",
            "value": 6.20052707951,
            "unit": "iter/sec",
            "range": "stddev: 0.00357659886488",
            "extra": "mean: 161.276612 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-128]",
            "value": 23.2449312035,
            "unit": "iter/sec",
            "range": "stddev: 0.000379430260939",
            "extra": "mean: 43.0201316255 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-256]",
            "value": 11.619178688,
            "unit": "iter/sec",
            "range": "stddev: 0.000407399810999",
            "extra": "mean: 86.0646029165 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-128]",
            "value": 23.0998618378,
            "unit": "iter/sec",
            "range": "stddev: 0.00011006373901",
            "extra": "mean: 43.2903022114 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-256]",
            "value": 11.5337131198,
            "unit": "iter/sec",
            "range": "stddev: 0.000649639906839",
            "extra": "mean: 86.7023472504 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-128]",
            "value": 27.1079053754,
            "unit": "iter/sec",
            "range": "stddev: 0.000182721623179",
            "extra": "mean: 36.8896078893 msec\nrounds: 27"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-256]",
            "value": 13.4648058774,
            "unit": "iter/sec",
            "range": "stddev: 0.000323555082161",
            "extra": "mean: 74.2676878602 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-128]",
            "value": 27.1310537884,
            "unit": "iter/sec",
            "range": "stddev: 0.00031550170352",
            "extra": "mean: 36.8581334069 msec\nrounds: 27"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-256]",
            "value": 13.4475481886,
            "unit": "iter/sec",
            "range": "stddev: 0.000378452595559",
            "extra": "mean: 74.3629980706 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_mask_many_leaves[cuda]",
            "value": 6486.7379747,
            "unit": "iter/sec",
            "range": "stddev: 0.0000157802921492",
            "extra": "mean: 154.160689687 usec\nrounds: 42"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-None]",
            "value": 6620.99057007,
            "unit": "iter/sec",
            "range": "stddev: 0.00000567399827347",
            "extra": "mean: 151.034802031 usec\nrounds: 1403"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-cpu]",
            "value": 6508.80570703,
            "unit": "iter/sec",
            "range": "stddev: 0.00000633203800913",
            "extra": "mean: 153.638016713 usec\nrounds: 4918"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to",
            "value": 23445.4295714,
            "unit": "iter/sec",
            "range": "stddev: 0.00000281648377102",
            "extra": "mean: 42.6522362047 usec\nrounds: 9881"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_nonblocking",
            "value": 23879.6653293,
            "unit": "iter/sec",
            "range": "stddev: 0.00000263689201515",
            "extra": "mean: 41.876633789 usec\nrounds: 10992"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[add]",
            "value": 10854.8783184,
            "unit": "iter/sec",
            "range": "stddev: 0.00000660137929932",
            "extra": "mean: 92.1244781073 usec\nrounds: 186"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[sub]",
            "value": 11078.2490013,
            "unit": "iter/sec",
            "range": "stddev: 0.00000359148127537",
            "extra": "mean: 90.2669726852 usec\nrounds: 7948"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_blocking",
            "value": 55.7055638989,
            "unit": "iter/sec",
            "range": "stddev: 0.000155401696353",
            "extra": "mean: 17.9515281779 msec\nrounds: 56"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_global_sync",
            "value": 94.6176963082,
            "unit": "iter/sec",
            "range": "stddev: 0.0119585293434",
            "extra": "mean: 10.5688474674 msec\nrounds: 107"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_event_sync",
            "value": 96.2495585708,
            "unit": "iter/sec",
            "range": "stddev: 0.0000955007928046",
            "extra": "mean: 10.3896580395 msec\nrounds: 97"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_default",
            "value": 94.2669066171,
            "unit": "iter/sec",
            "range": "stddev: 0.000108905393608",
            "extra": "mean: 10.608176675 msec\nrounds: 83"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_cpu_to_cpu_default",
            "value": 3906.02456011,
            "unit": "iter/sec",
            "range": "stddev: 0.00000533606961999",
            "extra": "mean: 256.014775281 usec\nrounds: 3779"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[False-None]",
            "value": 293.535642278,
            "unit": "iter/sec",
            "range": "stddev: 0.0000592430115554",
            "extra": "mean: 3.40674131509 msec\nrounds: 292"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[default-None]",
            "value": 445.675089528,
            "unit": "iter/sec",
            "range": "stddev: 0.000205336539648",
            "extra": "mean: 2.24378706259 msec\nrounds: 447"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[reduce-overhead-None]",
            "value": 444.547973275,
            "unit": "iter/sec",
            "range": "stddev: 0.000264634121254",
            "extra": "mean: 2.24947600736 msec\nrounds: 424"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[False-False-None]",
            "value": 494.591490215,
            "unit": "iter/sec",
            "range": "stddev: 0.0000325679766581",
            "extra": "mean: 2.021870614 msec\nrounds: 484"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-False-None]",
            "value": 617.926750536,
            "unit": "iter/sec",
            "range": "stddev: 0.0000415544740714",
            "extra": "mean: 1.61831479076 msec\nrounds: 611"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[within-False-None]",
            "value": 193.294316862,
            "unit": "iter/sec",
            "range": "stddev: 0.0000417146835358",
            "extra": "mean: 5.17345784518 msec\nrounds: 193"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-default-None]",
            "value": 116.210768098,
            "unit": "iter/sec",
            "range": "stddev: 0.000106954945627",
            "extra": "mean: 8.60505456048 msec\nrounds: 116"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[False-False-None]",
            "value": 142.440600968,
            "unit": "iter/sec",
            "range": "stddev: 0.0000647439504826",
            "extra": "mean: 7.02047023955 msec\nrounds: 142"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[True-False-None]",
            "value": 259.908999733,
            "unit": "iter/sec",
            "range": "stddev: 0.000086174187483",
            "extra": "mean: 3.84750047527 msec\nrounds: 261"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[within-False-None]",
            "value": 111.214830354,
            "unit": "iter/sec",
            "range": "stddev: 0.000095818620891",
            "extra": "mean: 8.9916065764 msec\nrounds: 111"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 12.3275163802,
            "unit": "iter/sec",
            "range": "stddev: 0.00124700457388",
            "extra": "mean: 81.1193406 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_archive",
            "value": 4.49576422441,
            "unit": "iter/sec",
            "range": "stddev: 0.00266210990743",
            "extra": "mean: 222.431593403 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_zarr",
            "value": 0.521264862341,
            "unit": "iter/sec",
            "range": "stddev: 0.00571483685932",
            "extra": "mean: 1.9184105284 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 43.0613298364,
            "unit": "iter/sec",
            "range": "stddev: 0.000409993594205",
            "extra": "mean: 23.2226920023 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 10.4742481913,
            "unit": "iter/sec",
            "range": "stddev: 0.0322971768493",
            "extra": "mean: 95.472246001 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 11.7805368169,
            "unit": "iter/sec",
            "range": "stddev: 0.000743059737308",
            "extra": "mean: 84.8857752018 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 43.8088911399,
            "unit": "iter/sec",
            "range": "stddev: 0.000588694908302",
            "extra": "mean: 22.8264166013 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-compile]",
            "value": 4796.8839393,
            "unit": "iter/sec",
            "range": "stddev: 0.0000136388829876",
            "extra": "mean: 208.468666879 usec\nrounds: 1742"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-eager]",
            "value": 8113.11959557,
            "unit": "iter/sec",
            "range": "stddev: 0.00000794658981788",
            "extra": "mean: 123.257150128 usec\nrounds: 6264"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-compile]",
            "value": 6512.4049773,
            "unit": "iter/sec",
            "range": "stddev: 0.00000651558625153",
            "extra": "mean: 153.553104189 usec\nrounds: 2798"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-eager]",
            "value": 5709.05267518,
            "unit": "iter/sec",
            "range": "stddev: 0.0000194854173614",
            "extra": "mean: 175.160408722 usec\nrounds: 5218"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-compile]",
            "value": 21143.1881751,
            "unit": "iter/sec",
            "range": "stddev: 0.00000319011818338",
            "extra": "mean: 47.2965567784 usec\nrounds: 15620"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-eager]",
            "value": 30552.7845594,
            "unit": "iter/sec",
            "range": "stddev: 0.00000123301451609",
            "extra": "mean: 32.7302409395 usec\nrounds: 26329"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-compile]",
            "value": 52429.861654,
            "unit": "iter/sec",
            "range": "stddev: 0.00000372978166047",
            "extra": "mean: 19.0731001085 usec\nrounds: 37120"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-eager]",
            "value": 15719.1155802,
            "unit": "iter/sec",
            "range": "stddev: 0.0000112533475629",
            "extra": "mean: 63.6168106847 usec\nrounds: 14691"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-compile]",
            "value": 3748.02346859,
            "unit": "iter/sec",
            "range": "stddev: 0.00000861663401311",
            "extra": "mean: 266.807294133 usec\nrounds: 1333"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-eager]",
            "value": 5367.93372802,
            "unit": "iter/sec",
            "range": "stddev: 0.0000061428603955",
            "extra": "mean: 186.291420622 usec\nrounds: 5016"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-compile]",
            "value": 5727.89474697,
            "unit": "iter/sec",
            "range": "stddev: 0.00000930813283818",
            "extra": "mean: 174.584213603 usec\nrounds: 3377"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-eager]",
            "value": 15654.6113457,
            "unit": "iter/sec",
            "range": "stddev: 0.00000470627899695",
            "extra": "mean: 63.8789413493 usec\nrounds: 13765"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-compile]",
            "value": 4115.65383834,
            "unit": "iter/sec",
            "range": "stddev: 0.00000959674041765",
            "extra": "mean: 242.974759122 usec\nrounds: 1375"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-eager]",
            "value": 1928.69299501,
            "unit": "iter/sec",
            "range": "stddev: 0.0000272796546176",
            "extra": "mean: 518.485836049 usec\nrounds: 1866"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-eager]",
            "value": 4380.51841289,
            "unit": "iter/sec",
            "range": "stddev: 0.0000127246388511",
            "extra": "mean: 228.283482854 usec\nrounds: 4166"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-compile]",
            "value": 3633.7639055,
            "unit": "iter/sec",
            "range": "stddev: 0.0000124237080778",
            "extra": "mean: 275.196745305 usec\nrounds: 1339"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-eager]",
            "value": 12956.9432195,
            "unit": "iter/sec",
            "range": "stddev: 0.00000603426267351",
            "extra": "mean: 77.1786974027 usec\nrounds: 11740"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-compile]",
            "value": 5852.43896898,
            "unit": "iter/sec",
            "range": "stddev: 0.00000569852861643",
            "extra": "mean: 170.868932645 usec\nrounds: 3286"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-eager]",
            "value": 2156.27213962,
            "unit": "iter/sec",
            "range": "stddev: 0.0000159512566074",
            "extra": "mean: 463.763354183 usec\nrounds: 2084"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-compile]",
            "value": 4156.64280982,
            "unit": "iter/sec",
            "range": "stddev: 0.0000098840019794",
            "extra": "mean: 240.578766508 usec\nrounds: 1410"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-compile]",
            "value": 31687.2243862,
            "unit": "iter/sec",
            "range": "stddev: 0.00000272172774056",
            "extra": "mean: 31.5584598959 usec\nrounds: 22026"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-eager]",
            "value": 54913.4279806,
            "unit": "iter/sec",
            "range": "stddev: 0.00000136186046252",
            "extra": "mean: 18.2104821494 usec\nrounds: 47079"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-compile]",
            "value": 46115.41529,
            "unit": "iter/sec",
            "range": "stddev: 0.00000231444260486",
            "extra": "mean: 21.684722857 usec\nrounds: 28886"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-eager]",
            "value": 19904.6098386,
            "unit": "iter/sec",
            "range": "stddev: 0.000011210290087",
            "extra": "mean: 50.2396182648 usec\nrounds: 17532"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-compile]",
            "value": 4215.33078031,
            "unit": "iter/sec",
            "range": "stddev: 0.000206376975095",
            "extra": "mean: 237.22930705 usec\nrounds: 98"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-eager]",
            "value": 307.233580912,
            "unit": "iter/sec",
            "range": "stddev: 0.0000639897582292",
            "extra": "mean: 3.25485253608 msec\nrounds: 302"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-compile]",
            "value": 5007.44470263,
            "unit": "iter/sec",
            "range": "stddev: 0.000215200097173",
            "extra": "mean: 199.702654624 usec\nrounds: 84"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-eager]",
            "value": 355.529696159,
            "unit": "iter/sec",
            "range": "stddev: 0.0000746411630195",
            "extra": "mean: 2.81270456674 msec\nrounds: 353"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-compile]",
            "value": 6231.60901072,
            "unit": "iter/sec",
            "range": "stddev: 0.00000635262972782",
            "extra": "mean: 160.472198798 usec\nrounds: 3896"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-eager]",
            "value": 18569.5695187,
            "unit": "iter/sec",
            "range": "stddev: 0.0000122708414526",
            "extra": "mean: 53.8515445387 usec\nrounds: 15868"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-compile]",
            "value": 7559.82135915,
            "unit": "iter/sec",
            "range": "stddev: 0.000013889872104",
            "extra": "mean: 132.278257976 usec\nrounds: 4459"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-eager]",
            "value": 22136.1691038,
            "unit": "iter/sec",
            "range": "stddev: 0.0000116365829638",
            "extra": "mean: 45.1749349814 usec\nrounds: 18235"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-compile]",
            "value": 7688.2415804,
            "unit": "iter/sec",
            "range": "stddev: 0.00000456169064412",
            "extra": "mean: 130.068753634 usec\nrounds: 7116"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-eager]",
            "value": 22284.3292661,
            "unit": "iter/sec",
            "range": "stddev: 0.0000111321387045",
            "extra": "mean: 44.8745837516 usec\nrounds: 18440"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-compile]",
            "value": 11303.4995876,
            "unit": "iter/sec",
            "range": "stddev: 0.00000417888132079",
            "extra": "mean: 88.4681768019 usec\nrounds: 9266"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-eager]",
            "value": 45486.5062526,
            "unit": "iter/sec",
            "range": "stddev: 0.0000108822806729",
            "extra": "mean: 21.9845418429 usec\nrounds: 33222"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-compile]",
            "value": 15799.7224838,
            "unit": "iter/sec",
            "range": "stddev: 0.00000345868953239",
            "extra": "mean: 63.2922509255 usec\nrounds: 12804"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-eager]",
            "value": 46694.8768215,
            "unit": "iter/sec",
            "range": "stddev: 0.000010745827407",
            "extra": "mean: 21.4156256118 usec\nrounds: 36562"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-compile]",
            "value": 15287.0017806,
            "unit": "iter/sec",
            "range": "stddev: 0.0000038350225695",
            "extra": "mean: 65.4150509269 usec\nrounds: 13261"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-eager]",
            "value": 46588.1468896,
            "unit": "iter/sec",
            "range": "stddev: 0.0000110955839142",
            "extra": "mean: 21.4646871954 usec\nrounds: 36257"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-compile]",
            "value": 11585.8244274,
            "unit": "iter/sec",
            "range": "stddev: 0.00000627409377699",
            "extra": "mean: 86.3123730443 usec\nrounds: 10328"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-eager]",
            "value": 45586.8509431,
            "unit": "iter/sec",
            "range": "stddev: 0.0000107515048291",
            "extra": "mean: 21.9361499931 usec\nrounds: 37258"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-compile]",
            "value": 14809.8824354,
            "unit": "iter/sec",
            "range": "stddev: 0.00000767940299606",
            "extra": "mean: 67.5224806384 usec\nrounds: 12812"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-eager]",
            "value": 39902.8534066,
            "unit": "iter/sec",
            "range": "stddev: 0.0000112265310885",
            "extra": "mean: 25.0608644402 usec\nrounds: 37508"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-compile]",
            "value": 15522.4961523,
            "unit": "iter/sec",
            "range": "stddev: 0.00000314603113927",
            "extra": "mean: 64.4226283061 usec\nrounds: 13459"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-eager]",
            "value": 47072.4820352,
            "unit": "iter/sec",
            "range": "stddev: 0.0000106281016497",
            "extra": "mean: 21.2438341206 usec\nrounds: 37508"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-eager]",
            "value": 32530.5163619,
            "unit": "iter/sec",
            "range": "stddev: 0.00000230704501728",
            "extra": "mean: 30.7403666414 usec\nrounds: 23861"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-compile]",
            "value": 6969.15371773,
            "unit": "iter/sec",
            "range": "stddev: 0.00000469973116386",
            "extra": "mean: 143.489445132 usec\nrounds: 4340"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-eager]",
            "value": 2707.87738141,
            "unit": "iter/sec",
            "range": "stddev: 0.0000150312241888",
            "extra": "mean: 369.292940243 usec\nrounds: 2569"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-compile]",
            "value": 6557.42037787,
            "unit": "iter/sec",
            "range": "stddev: 0.00000702635771579",
            "extra": "mean: 152.498992344 usec\nrounds: 3597"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[eager]",
            "value": 6052.88654269,
            "unit": "iter/sec",
            "range": "stddev: 0.00000674648365674",
            "extra": "mean: 165.210431907 usec\nrounds: 5820"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[compile]",
            "value": 5826.51429667,
            "unit": "iter/sec",
            "range": "stddev: 0.00000531109253391",
            "extra": "mean: 171.629202141 usec\nrounds: 3888"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[eager]",
            "value": 5643.75217546,
            "unit": "iter/sec",
            "range": "stddev: 0.00000815851377127",
            "extra": "mean: 177.187085632 usec\nrounds: 5388"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[compile]",
            "value": 5207.28151333,
            "unit": "iter/sec",
            "range": "stddev: 0.0000127486326504",
            "extra": "mean: 192.038782124 usec\nrounds: 3561"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-eager]",
            "value": 108060.96994,
            "unit": "iter/sec",
            "range": "stddev: 0.00000127035275695",
            "extra": "mean: 9.25403501891 usec\nrounds: 81500"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-compile]",
            "value": 42873.388775,
            "unit": "iter/sec",
            "range": "stddev: 0.00000224922815141",
            "extra": "mean: 23.3244916852 usec\nrounds: 26560"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-eager]",
            "value": 65544.2475342,
            "unit": "iter/sec",
            "range": "stddev: 0.00000127350298431",
            "extra": "mean: 15.2568690254 usec\nrounds: 55711"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-compile]",
            "value": 33589.9332379,
            "unit": "iter/sec",
            "range": "stddev: 0.00000586935646094",
            "extra": "mean: 29.7708242799 usec\nrounds: 23369"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-eager]",
            "value": 36447.9840766,
            "unit": "iter/sec",
            "range": "stddev: 0.00000191333496737",
            "extra": "mean: 27.4363596598 usec\nrounds: 32993"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-compile]",
            "value": 25962.7224611,
            "unit": "iter/sec",
            "range": "stddev: 0.00000311762586278",
            "extra": "mean: 38.5167619266 usec\nrounds: 18629"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[eager]",
            "value": 36950.0992363,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020429048217",
            "extra": "mean: 27.0635267744 usec\nrounds: 29498"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[compile]",
            "value": 8621.39692261,
            "unit": "iter/sec",
            "range": "stddev: 0.00000578408336264",
            "extra": "mean: 115.99048379 usec\nrounds: 7392"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[eager]",
            "value": 30783.0801901,
            "unit": "iter/sec",
            "range": "stddev: 0.00000309723812656",
            "extra": "mean: 32.4853781306 usec\nrounds: 21547"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile]",
            "value": 8993.80030587,
            "unit": "iter/sec",
            "range": "stddev: 0.0000146291221413",
            "extra": "mean: 111.187703306 usec\nrounds: 5560"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile-overhead]",
            "value": 5197.6468739,
            "unit": "iter/sec",
            "range": "stddev: 0.0000172303301169",
            "extra": "mean: 192.3947556 usec\nrounds: 307"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[eager]",
            "value": 3874.06865807,
            "unit": "iter/sec",
            "range": "stddev: 0.00000693180915554",
            "extra": "mean: 258.126555893 usec\nrounds: 3528"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile]",
            "value": 3165.01014786,
            "unit": "iter/sec",
            "range": "stddev: 0.0000112341639944",
            "extra": "mean: 315.954753155 usec\nrounds: 2498"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile-overhead]",
            "value": 251.897449397,
            "unit": "iter/sec",
            "range": "stddev: 0.00180162542725",
            "extra": "mean: 3.96986949408 msec\nrounds: 1028"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[False]",
            "value": 6302.08999,
            "unit": "iter/sec",
            "range": "stddev: 0.00000578481017242",
            "extra": "mean: 158.677518345 usec\nrounds: 5223"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[True]",
            "value": 6208.65126135,
            "unit": "iter/sec",
            "range": "stddev: 0.00000653265370503",
            "extra": "mean: 161.065577354 usec\nrounds: 5348"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[eager]",
            "value": 674.473165434,
            "unit": "iter/sec",
            "range": "stddev: 0.0000280584409966",
            "extra": "mean: 1.48263867452 msec\nrounds: 651"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile]",
            "value": 761.337287035,
            "unit": "iter/sec",
            "range": "stddev: 0.0000287046055372",
            "extra": "mean: 1.31347829277 msec\nrounds: 608"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile-overhead]",
            "value": 1036.54329454,
            "unit": "iter/sec",
            "range": "stddev: 0.0000996328569078",
            "extra": "mean: 964.745037925 usec\nrounds: 79"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[eager]",
            "value": 10265.3315554,
            "unit": "iter/sec",
            "range": "stddev: 0.0000100136518414",
            "extra": "mean: 97.4152656054 usec\nrounds: 8713"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile]",
            "value": 7866.11447595,
            "unit": "iter/sec",
            "range": "stddev: 0.0000103571426983",
            "extra": "mean: 127.127567627 usec\nrounds: 6961"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile-overhead]",
            "value": 4948.01394383,
            "unit": "iter/sec",
            "range": "stddev: 0.0000124866310611",
            "extra": "mean: 202.101289801 usec\nrounds: 3250"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[eager]",
            "value": 2522.71167751,
            "unit": "iter/sec",
            "range": "stddev: 0.00000850437395292",
            "extra": "mean: 396.39884689 usec\nrounds: 2430"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile]",
            "value": 2955.54777154,
            "unit": "iter/sec",
            "range": "stddev: 0.0000130350065014",
            "extra": "mean: 338.346755761 usec\nrounds: 2755"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile-overhead]",
            "value": 3283.84862944,
            "unit": "iter/sec",
            "range": "stddev: 0.0000125672163592",
            "extra": "mean: 304.520735528 usec\nrounds: 2189"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-eager]",
            "value": 1198.23809297,
            "unit": "iter/sec",
            "range": "stddev: 0.0000282186826199",
            "extra": "mean: 834.558679002 usec\nrounds: 1225"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile]",
            "value": 1428.69294011,
            "unit": "iter/sec",
            "range": "stddev: 0.0000130852161689",
            "extra": "mean: 699.940464412 usec\nrounds: 1363"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile-overhead]",
            "value": 2128.7418768,
            "unit": "iter/sec",
            "range": "stddev: 0.000028684873906",
            "extra": "mean: 469.761040969 usec\nrounds: 1344"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-eager]",
            "value": 792.67732537,
            "unit": "iter/sec",
            "range": "stddev: 0.0000172198383495",
            "extra": "mean: 1.26154737621 msec\nrounds: 771"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile]",
            "value": 1371.28434122,
            "unit": "iter/sec",
            "range": "stddev: 0.0000230700396932",
            "extra": "mean: 729.243359631 usec\nrounds: 1321"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile-overhead]",
            "value": 2020.2752228,
            "unit": "iter/sec",
            "range": "stddev: 0.000010582294999",
            "extra": "mean: 494.982064183 usec\nrounds: 1287"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-eager]",
            "value": 1190.94615209,
            "unit": "iter/sec",
            "range": "stddev: 0.0000392173329104",
            "extra": "mean: 839.668525938 usec\nrounds: 1245"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile]",
            "value": 1423.4648263,
            "unit": "iter/sec",
            "range": "stddev: 0.0000239651662574",
            "extra": "mean: 702.511211742 usec\nrounds: 1384"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile-overhead]",
            "value": 2220.27557971,
            "unit": "iter/sec",
            "range": "stddev: 0.000010401579331",
            "extra": "mean: 450.394540723 usec\nrounds: 1115"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-eager]",
            "value": 762.163899506,
            "unit": "iter/sec",
            "range": "stddev: 0.0000206534474431",
            "extra": "mean: 1.31205374677 msec\nrounds: 746"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile]",
            "value": 1235.40295824,
            "unit": "iter/sec",
            "range": "stddev: 0.000014246313464",
            "extra": "mean: 809.452489433 usec\nrounds: 1185"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile-overhead]",
            "value": 1319.65106486,
            "unit": "iter/sec",
            "range": "stddev: 0.0000643241984386",
            "extra": "mean: 757.776071741 usec\nrounds: 1022"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[eager]",
            "value": 349.216475535,
            "unit": "iter/sec",
            "range": "stddev: 0.0000452457251859",
            "extra": "mean: 2.86355332596 msec\nrounds: 344"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile]",
            "value": 1341.44613404,
            "unit": "iter/sec",
            "range": "stddev: 0.0000121669706776",
            "extra": "mean: 745.464148445 usec\nrounds: 1002"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile-overhead]",
            "value": 1929.42793257,
            "unit": "iter/sec",
            "range": "stddev: 0.0000373485092156",
            "extra": "mean: 518.288339835 usec\nrounds: 127"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1043.14723973,
            "unit": "iter/sec",
            "range": "stddev: 0.0000410718949103",
            "extra": "mean: 958.637440536 usec\nrounds: 962"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1038.18549525,
            "unit": "iter/sec",
            "range": "stddev: 0.00004026799512",
            "extra": "mean: 963.219005253 usec\nrounds: 951"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1153.84409342,
            "unit": "iter/sec",
            "range": "stddev: 0.0000367929331195",
            "extra": "mean: 866.668214275 usec\nrounds: 1134"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1152.99681134,
            "unit": "iter/sec",
            "range": "stddev: 0.0000365231337603",
            "extra": "mean: 867.305087203 usec\nrounds: 1148"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-True]",
            "value": 37.6461656469,
            "unit": "iter/sec",
            "range": "stddev: 0.000498391216648",
            "extra": "mean: 26.5631302104 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-False]",
            "value": 37.07618639,
            "unit": "iter/sec",
            "range": "stddev: 0.000701575396081",
            "extra": "mean: 26.9714902574 msec\nrounds: 39"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-True]",
            "value": 36.9466677446,
            "unit": "iter/sec",
            "range": "stddev: 0.00155493857126",
            "extra": "mean: 27.0660403507 msec\nrounds: 37"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-False]",
            "value": 37.6716736219,
            "unit": "iter/sec",
            "range": "stddev: 0.00055907659452",
            "extra": "mean: 26.5451439731 msec\nrounds: 38"
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
          "id": "04d2b4e4073486ef8945d7c648ddb16a47c96ac4",
          "message": "[Performance] Skip the identity movedim in pack along the last dim (#3070)",
          "timestamp": "2026-10-20T05:18:40-04:00",
          "tree_id": "777a85af146e8353a1db7cc26ccc088c7010ba48",
          "url": "https://github.com/pytorch/tensordict/commit/04d2b4e4073486ef8945d7c648ddb16a47c96ac4"
        },
        "date": 1792488561014,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-2]",
            "value": 4550.36709843,
            "unit": "iter/sec",
            "range": "stddev: 0.00000681609085221",
            "extra": "mean: 219.762489129 usec\nrounds: 137"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-8]",
            "value": 3577.80551026,
            "unit": "iter/sec",
            "range": "stddev: 0.00000774046741208",
            "extra": "mean: 279.50093909 usec\nrounds: 3089"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-32]",
            "value": 2024.32338412,
            "unit": "iter/sec",
            "range": "stddev: 0.0000220124695427",
            "extra": "mean: 493.992218754 usec\nrounds: 1865"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-2]",
            "value": 4458.54528611,
            "unit": "iter/sec",
            "range": "stddev: 0.00000817671562704",
            "extra": "mean: 224.288402568 usec\nrounds: 4035"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-8]",
            "value": 3566.33044069,
            "unit": "iter/sec",
            "range": "stddev: 0.00000756660877237",
            "extra": "mean: 280.400264818 usec\nrounds: 3215"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-32]",
            "value": 1797.27640427,
            "unit": "iter/sec",
            "range": "stddev: 0.0000109171183699",
            "extra": "mean: 556.397445391 usec\nrounds: 1655"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-2]",
            "value": 2294.07235917,
            "unit": "iter/sec",
            "range": "stddev: 0.00000636798302399",
            "extra": "mean: 435.906041064 usec\nrounds: 73"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-8]",
            "value": 585.64688703,
            "unit": "iter/sec",
            "range": "stddev: 0.0000433487389217",
            "extra": "mean: 1.70751355833 msec\nrounds: 600"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-32]",
            "value": 151.932937186,
            "unit": "iter/sec",
            "range": "stddev: 0.000171729004693",
            "extra": "mean: 6.58185129913 msec\nrounds: 154"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-2]",
            "value": 2307.18796539,
            "unit": "iter/sec",
            "range": "stddev: 0.0000114742122709",
            "extra": "mean: 433.428058313 usec\nrounds: 2126"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-8]",
            "value": 611.339440691,
            "unit": "iter/sec",
            "range": "stddev: 0.0000208350041406",
            "extra": "mean: 1.63575246981 msec\nrounds: 592"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-32]",
            "value": 155.438020955,
            "unit": "iter/sec",
            "range": "stddev: 0.0000676574203389",
            "extra": "mean: 6.43343239871 msec\nrounds: 153"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-2]",
            "value": 3138.177942,
            "unit": "iter/sec",
            "range": "stddev: 0.00000696044461313",
            "extra": "mean: 318.656245274 usec\nrounds: 2406"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-8]",
            "value": 844.261547114,
            "unit": "iter/sec",
            "range": "stddev: 0.0000264583734262",
            "extra": "mean: 1.18446706879 msec\nrounds: 814"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-32]",
            "value": 216.834954181,
            "unit": "iter/sec",
            "range": "stddev: 0.0000426228717111",
            "extra": "mean: 4.6118025748 msec\nrounds: 214"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-2]",
            "value": 3124.59066343,
            "unit": "iter/sec",
            "range": "stddev: 0.00000672117264128",
            "extra": "mean: 320.041921556 usec\nrounds: 2792"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-8]",
            "value": 838.486352985,
            "unit": "iter/sec",
            "range": "stddev: 0.000012792644333",
            "extra": "mean: 1.19262525435 msec\nrounds: 825"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-32]",
            "value": 214.864240017,
            "unit": "iter/sec",
            "range": "stddev: 0.0000284137764628",
            "extra": "mean: 4.65410158489 msec\nrounds: 212"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-2]",
            "value": 3649.23481801,
            "unit": "iter/sec",
            "range": "stddev: 0.00000632041345462",
            "extra": "mean: 274.030050099 usec\nrounds: 3012"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-8]",
            "value": 984.49932407,
            "unit": "iter/sec",
            "range": "stddev: 0.0000120994931195",
            "extra": "mean: 1.01574472989 msec\nrounds: 955"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-32]",
            "value": 249.867956372,
            "unit": "iter/sec",
            "range": "stddev: 0.0000439506500416",
            "extra": "mean: 4.00211381451 msec\nrounds: 248"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-2]",
            "value": 3627.8314084,
            "unit": "iter/sec",
            "range": "stddev: 0.0000131431191446",
            "extra": "mean: 275.64676729 usec\nrounds: 3326"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-8]",
            "value": 972.009114831,
            "unit": "iter/sec",
            "range": "stddev: 0.0000168477588887",
            "extra": "mean: 1.02879693692 msec\nrounds: 951"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-32]",
            "value": 250.137126821,
            "unit": "iter/sec",
            "range": "stddev: 0.0000870659302447",
            "extra": "mean: 3.99780717364 msec\nrounds: 248"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-128]",
            "value": 189.947720621,
            "unit": "iter/sec",
            "range": "stddev: 0.000110254650097",
            "extra": "mean: 5.26460647556 msec\nrounds: 185"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-256]",
            "value": 99.2080290583,
            "unit": "iter/sec",
            "range": "stddev: 0.000113328485338",
            "extra": "mean: 10.0798293192 msec\nrounds: 97"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-128]",
            "value": 181.880079943,
            "unit": "iter/sec",
            "range": "stddev: 0.0000341992766839",
            "extra": "mean: 5.49812821894 msec\nrounds: 178"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-256]",
            "value": 94.7821377748,
            "unit": "iter/sec",
            "range": "stddev: 0.0000474400148257",
            "extra": "mean: 10.5505111351 msec\nrounds: 89"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-128]",
            "value": 13.2172486427,
            "unit": "iter/sec",
            "range": "stddev: 0.000399768413314",
            "extra": "mean: 75.6587113579 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-256]",
            "value": 6.61273487,
            "unit": "iter/sec",
            "range": "stddev: 0.000617392149937",
            "extra": "mean: 151.223362143 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-128]",
            "value": 12.8486816606,
            "unit": "iter/sec",
            "range": "stddev: 0.00176145129815",
            "extra": "mean: 77.8289965009 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-256]",
            "value": 6.16890858401,
            "unit": "iter/sec",
            "range": "stddev: 0.00409202445109",
            "extra": "mean: 162.103228858 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-128]",
            "value": 23.6446042193,
            "unit": "iter/sec",
            "range": "stddev: 0.000550418212301",
            "extra": "mean: 42.2929472925 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-256]",
            "value": 11.8655903891,
            "unit": "iter/sec",
            "range": "stddev: 0.000568440839962",
            "extra": "mean: 84.2773066664 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-128]",
            "value": 23.6023072354,
            "unit": "iter/sec",
            "range": "stddev: 0.000338748120289",
            "extra": "mean: 42.3687392095 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-256]",
            "value": 11.841117484,
            "unit": "iter/sec",
            "range": "stddev: 0.000187188181155",
            "extra": "mean: 84.451488751 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-128]",
            "value": 27.7537784142,
            "unit": "iter/sec",
            "range": "stddev: 0.000658015440098",
            "extra": "mean: 36.0311300709 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-256]",
            "value": 13.6985427721,
            "unit": "iter/sec",
            "range": "stddev: 0.000707617788997",
            "extra": "mean: 73.0004655707 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-128]",
            "value": 27.7527898896,
            "unit": "iter/sec",
            "range": "stddev: 0.000527450655445",
            "extra": "mean: 36.032413461 msec\nrounds: 26"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-256]",
            "value": 13.7421067878,
            "unit": "iter/sec",
            "range": "stddev: 0.000379179005229",
            "extra": "mean: 72.769045929 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_mask_many_leaves[cuda]",
            "value": 6533.18051328,
            "unit": "iter/sec",
            "range": "stddev: 0.0000156735419757",
            "extra": "mean: 153.064804802 usec\nrounds: 41"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-None]",
            "value": 6566.04080521,
            "unit": "iter/sec",
            "range": "stddev: 0.00000739000249656",
            "extra": "mean: 152.298779381 usec\nrounds: 1383"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-cpu]",
            "value": 6601.5431953,
            "unit": "iter/sec",
            "range": "stddev: 0.00000666594011817",
            "extra": "mean: 151.479732907 usec\nrounds: 4837"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to",
            "value": 22504.3951277,
            "unit": "iter/sec",
            "range": "stddev: 0.000003270564048",
            "extra": "mean: 44.4357644063 usec\nrounds: 11254"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_nonblocking",
            "value": 22455.4214119,
            "unit": "iter/sec",
            "range": "stddev: 0.00000326987552865",
            "extra": "mean: 44.5326757248 usec\nrounds: 11971"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[add]",
            "value": 10374.8227004,
            "unit": "iter/sec",
            "range": "stddev: 0.00000895288639925",
            "extra": "mean: 96.3871893403 usec\nrounds: 174"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[sub]",
            "value": 10724.317327,
            "unit": "iter/sec",
            "range": "stddev: 0.00000499828917486",
            "extra": "mean: 93.2460285828 usec\nrounds: 8045"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_blocking",
            "value": 41.2899560018,
            "unit": "iter/sec",
            "range": "stddev: 0.000787572491991",
            "extra": "mean: 24.2189650179 msec\nrounds: 58"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_global_sync",
            "value": 96.1020015061,
            "unit": "iter/sec",
            "range": "stddev: 0.0120022275212",
            "extra": "mean: 10.4056105422 msec\nrounds: 107"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_event_sync",
            "value": 96.8904560318,
            "unit": "iter/sec",
            "range": "stddev: 0.000286052679359",
            "extra": "mean: 10.32093398 msec\nrounds: 99"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_default",
            "value": 95.3080494739,
            "unit": "iter/sec",
            "range": "stddev: 0.000252625042991",
            "extra": "mean: 10.4922932063 msec\nrounds: 97"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_cpu_to_cpu_default",
            "value": 3900.00172958,
            "unit": "iter/sec",
            "range": "stddev: 0.000010351308765",
            "extra": "mean: 256.410142697 usec\nrounds: 3925"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[False-None]",
            "value": 295.518948679,
            "unit": "iter/sec",
            "range": "stddev: 0.0000330817770075",
            "extra": "mean: 3.38387776645 msec\nrounds: 227"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[default-None]",
            "value": 449.336878896,
            "unit": "iter/sec",
            "range": "stddev: 0.00021863365981",
            "extra": "mean: 2.22550172703 msec\nrounds: 432"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[reduce-overhead-None]",
            "value": 447.358329735,
            "unit": "iter/sec",
            "range": "stddev: 0.000274315355533",
            "extra": "mean: 2.23534454045 msec\nrounds: 422"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[False-False-None]",
            "value": 495.63208068,
            "unit": "iter/sec",
            "range": "stddev: 0.000042883916087",
            "extra": "mean: 2.01762565213 msec\nrounds: 486"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-False-None]",
            "value": 630.597766148,
            "unit": "iter/sec",
            "range": "stddev: 0.000057552260163",
            "extra": "mean: 1.58579692743 msec\nrounds: 633"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[within-False-None]",
            "value": 192.313163001,
            "unit": "iter/sec",
            "range": "stddev: 0.000344643274331",
            "extra": "mean: 5.19985207666 msec\nrounds: 195"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-default-None]",
            "value": 116.589201015,
            "unit": "iter/sec",
            "range": "stddev: 0.0000901308931326",
            "extra": "mean: 8.57712370697 msec\nrounds: 116"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[False-False-None]",
            "value": 143.798969077,
            "unit": "iter/sec",
            "range": "stddev: 0.0000505127965162",
            "extra": "mean: 6.95415277605 msec\nrounds: 143"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[True-False-None]",
            "value": 267.377802631,
            "unit": "iter/sec",
            "range": "stddev: 0.0000471411850263",
            "extra": "mean: 3.7400262481 msec\nrounds: 262"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[within-False-None]",
            "value": 112.635969444,
            "unit": "iter/sec",
            "range": "stddev: 0.000110653802372",
            "extra": "mean: 8.87815859299 msec\nrounds: 113"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 10.8210603084,
            "unit": "iter/sec",
            "range": "stddev: 0.000498996916439",
            "extra": "mean: 92.4123858014 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_archive",
            "value": 4.58593190658,
            "unit": "iter/sec",
            "range": "stddev: 0.000903278574495",
            "extra": "mean: 218.058187599 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_zarr",
            "value": 0.455160965657,
            "unit": "iter/sec",
            "range": "stddev: 0.00704384488977",
            "extra": "mean: 2.1970249548 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 41.4336713202,
            "unit": "iter/sec",
            "range": "stddev: 0.000234040257415",
            "extra": "mean: 24.1349600008 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.62672387807,
            "unit": "iter/sec",
            "range": "stddev: 0.0276956617879",
            "extra": "mean: 103.877498998 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 10.0996114985,
            "unit": "iter/sec",
            "range": "stddev: 0.00209848311239",
            "extra": "mean: 99.0137096014 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 43.9771936047,
            "unit": "iter/sec",
            "range": "stddev: 0.000929060846755",
            "extra": "mean: 22.7390589993 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-compile]",
            "value": 4861.23462813,
            "unit": "iter/sec",
            "range": "stddev: 0.00000588067731002",
            "extra": "mean: 205.709058808 usec\nrounds: 1805"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-eager]",
            "value": 8201.30451313,
            "unit": "iter/sec",
            "range": "stddev: 0.00000524991840962",
            "extra": "mean: 121.931821749 usec\nrounds: 7099"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-compile]",
            "value": 6454.65258619,
            "unit": "iter/sec",
            "range": "stddev: 0.00000831788054948",
            "extra": "mean: 154.927006008 usec\nrounds: 1992"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-eager]",
            "value": 5749.09493332,
            "unit": "iter/sec",
            "range": "stddev: 0.0000180995512959",
            "extra": "mean: 173.940422205 usec\nrounds: 5377"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-compile]",
            "value": 21221.8796564,
            "unit": "iter/sec",
            "range": "stddev: 0.00000387601402628",
            "extra": "mean: 47.1211794709 usec\nrounds: 16176"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-eager]",
            "value": 31204.8415033,
            "unit": "iter/sec",
            "range": "stddev: 0.00000345384366687",
            "extra": "mean: 32.0463092208 usec\nrounds: 25754"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-compile]",
            "value": 55321.1292054,
            "unit": "iter/sec",
            "range": "stddev: 0.00000544781687888",
            "extra": "mean: 18.0762759973 usec\nrounds: 38241"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-eager]",
            "value": 15979.3683115,
            "unit": "iter/sec",
            "range": "stddev: 0.0000114022982979",
            "extra": "mean: 62.5806965897 usec\nrounds: 14571"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-compile]",
            "value": 3740.1742989,
            "unit": "iter/sec",
            "range": "stddev: 0.000015220458882",
            "extra": "mean: 267.367218767 usec\nrounds: 1285"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-eager]",
            "value": 5251.21800535,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113224728365",
            "extra": "mean: 190.432010056 usec\nrounds: 5081"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-compile]",
            "value": 5780.95026427,
            "unit": "iter/sec",
            "range": "stddev: 0.0000149862482456",
            "extra": "mean: 172.981941426 usec\nrounds: 3409"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-eager]",
            "value": 15762.5048515,
            "unit": "iter/sec",
            "range": "stddev: 0.00000532665621353",
            "extra": "mean: 63.4416933995 usec\nrounds: 13727"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-compile]",
            "value": 4103.51915285,
            "unit": "iter/sec",
            "range": "stddev: 0.0000123655894263",
            "extra": "mean: 243.693269789 usec\nrounds: 1375"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-eager]",
            "value": 1927.98329107,
            "unit": "iter/sec",
            "range": "stddev: 0.0000246309489895",
            "extra": "mean: 518.67669426 usec\nrounds: 1920"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-eager]",
            "value": 4352.05756787,
            "unit": "iter/sec",
            "range": "stddev: 0.00000953512621296",
            "extra": "mean: 229.776372303 usec\nrounds: 4051"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-compile]",
            "value": 3611.54611466,
            "unit": "iter/sec",
            "range": "stddev: 0.0000096236628981",
            "extra": "mean: 276.889722089 usec\nrounds: 1331"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-eager]",
            "value": 12861.204967,
            "unit": "iter/sec",
            "range": "stddev: 0.00000517781577217",
            "extra": "mean: 77.7532122818 usec\nrounds: 11471"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-compile]",
            "value": 5752.57366621,
            "unit": "iter/sec",
            "range": "stddev: 0.00000952213506537",
            "extra": "mean: 173.835235848 usec\nrounds: 3273"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-eager]",
            "value": 2183.79314102,
            "unit": "iter/sec",
            "range": "stddev: 0.0000163936804819",
            "extra": "mean: 457.918829954 usec\nrounds: 2140"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-compile]",
            "value": 4101.5193147,
            "unit": "iter/sec",
            "range": "stddev: 0.0000117562355247",
            "extra": "mean: 243.812090904 usec\nrounds: 1373"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-compile]",
            "value": 31357.8065941,
            "unit": "iter/sec",
            "range": "stddev: 0.00000220612782421",
            "extra": "mean: 31.8899855766 usec\nrounds: 22228"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-eager]",
            "value": 54567.4519139,
            "unit": "iter/sec",
            "range": "stddev: 0.00000136743978541",
            "extra": "mean: 18.3259427539 usec\nrounds: 45557"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-compile]",
            "value": 44916.3374804,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017854470368",
            "extra": "mean: 22.2636140009 usec\nrounds: 27980"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-eager]",
            "value": 19706.5721824,
            "unit": "iter/sec",
            "range": "stddev: 0.0000112464324781",
            "extra": "mean: 50.7444922814 usec\nrounds: 17111"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-compile]",
            "value": 4175.30105167,
            "unit": "iter/sec",
            "range": "stddev: 0.000211552865497",
            "extra": "mean: 239.503687908 usec\nrounds: 96"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-eager]",
            "value": 291.261595979,
            "unit": "iter/sec",
            "range": "stddev: 0.000175459718674",
            "extra": "mean: 3.43333969808 msec\nrounds: 285"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-compile]",
            "value": 5030.14326443,
            "unit": "iter/sec",
            "range": "stddev: 0.000208678849326",
            "extra": "mean: 198.801494795 usec\nrounds: 89"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-eager]",
            "value": 360.083481169,
            "unit": "iter/sec",
            "range": "stddev: 0.0000323146653026",
            "extra": "mean: 2.77713378229 msec\nrounds: 358"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-compile]",
            "value": 6306.47107379,
            "unit": "iter/sec",
            "range": "stddev: 0.00000705782937463",
            "extra": "mean: 158.567285618 usec\nrounds: 3938"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-eager]",
            "value": 19343.9527196,
            "unit": "iter/sec",
            "range": "stddev: 0.0000119606708418",
            "extra": "mean: 51.6957425658 usec\nrounds: 15655"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-compile]",
            "value": 7689.6084191,
            "unit": "iter/sec",
            "range": "stddev: 0.00000452655431561",
            "extra": "mean: 130.04563373 usec\nrounds: 4519"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-eager]",
            "value": 22747.9087676,
            "unit": "iter/sec",
            "range": "stddev: 0.0000114651359548",
            "extra": "mean: 43.9600848684 usec\nrounds: 18709"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-compile]",
            "value": 7646.86845989,
            "unit": "iter/sec",
            "range": "stddev: 0.00000421572549229",
            "extra": "mean: 130.772486181 usec\nrounds: 7016"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-eager]",
            "value": 22954.1370382,
            "unit": "iter/sec",
            "range": "stddev: 0.0000112118801633",
            "extra": "mean: 43.5651315637 usec\nrounds: 8802"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-compile]",
            "value": 12106.854498,
            "unit": "iter/sec",
            "range": "stddev: 0.00000354525940919",
            "extra": "mean: 82.5978374619 usec\nrounds: 9917"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-eager]",
            "value": 46913.4683717,
            "unit": "iter/sec",
            "range": "stddev: 0.0000106240989651",
            "extra": "mean: 21.3158403057 usec\nrounds: 34555"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-compile]",
            "value": 15762.781358,
            "unit": "iter/sec",
            "range": "stddev: 0.00000664015269706",
            "extra": "mean: 63.4405805225 usec\nrounds: 13144"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-eager]",
            "value": 45673.41428,
            "unit": "iter/sec",
            "range": "stddev: 0.0000117870392143",
            "extra": "mean: 21.8945751213 usec\nrounds: 34891"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-compile]",
            "value": 15583.4467857,
            "unit": "iter/sec",
            "range": "stddev: 0.00000533247228773",
            "extra": "mean: 64.170655809 usec\nrounds: 13635"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-eager]",
            "value": 44954.4197114,
            "unit": "iter/sec",
            "range": "stddev: 0.0000112563817391",
            "extra": "mean: 22.2447538289 usec\nrounds: 37328"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-compile]",
            "value": 11964.6252946,
            "unit": "iter/sec",
            "range": "stddev: 0.00000439733969029",
            "extra": "mean: 83.5797173232 usec\nrounds: 10250"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-eager]",
            "value": 46226.8213098,
            "unit": "iter/sec",
            "range": "stddev: 0.0000107852805779",
            "extra": "mean: 21.6324629656 usec\nrounds: 38670"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-compile]",
            "value": 15657.6566627,
            "unit": "iter/sec",
            "range": "stddev: 0.00000356066574151",
            "extra": "mean: 63.8665172922 usec\nrounds: 13526"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-eager]",
            "value": 47608.4148161,
            "unit": "iter/sec",
            "range": "stddev: 0.0000108125283725",
            "extra": "mean: 21.0046901134 usec\nrounds: 37552"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-compile]",
            "value": 15415.0577322,
            "unit": "iter/sec",
            "range": "stddev: 0.00000495109883648",
            "extra": "mean: 64.8716350837 usec\nrounds: 13626"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-eager]",
            "value": 47827.8888781,
            "unit": "iter/sec",
            "range": "stddev: 0.0000109822903389",
            "extra": "mean: 20.9083031565 usec\nrounds: 36192"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-eager]",
            "value": 33340.6569846,
            "unit": "iter/sec",
            "range": "stddev: 0.00000234885665868",
            "extra": "mean: 29.9934101617 usec\nrounds: 23579"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-compile]",
            "value": 6984.81046024,
            "unit": "iter/sec",
            "range": "stddev: 0.000100456556057",
            "extra": "mean: 143.167807587 usec\nrounds: 4396"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-eager]",
            "value": 2575.97029021,
            "unit": "iter/sec",
            "range": "stddev: 0.0000313246430699",
            "extra": "mean: 388.203235029 usec\nrounds: 2595"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-compile]",
            "value": 6571.31918963,
            "unit": "iter/sec",
            "range": "stddev: 0.00000622725924441",
            "extra": "mean: 152.176446029 usec\nrounds: 3610"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[eager]",
            "value": 6209.89457272,
            "unit": "iter/sec",
            "range": "stddev: 0.0000043239022251",
            "extra": "mean: 161.033329679 usec\nrounds: 5967"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[compile]",
            "value": 5838.30178223,
            "unit": "iter/sec",
            "range": "stddev: 0.00000539930546358",
            "extra": "mean: 171.282684126 usec\nrounds: 3800"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[eager]",
            "value": 5641.38669124,
            "unit": "iter/sec",
            "range": "stddev: 0.00000626101332916",
            "extra": "mean: 177.261381772 usec\nrounds: 5564"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[compile]",
            "value": 5304.9138889,
            "unit": "iter/sec",
            "range": "stddev: 0.00000901697766042",
            "extra": "mean: 188.504473577 usec\nrounds: 3651"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-eager]",
            "value": 112392.340279,
            "unit": "iter/sec",
            "range": "stddev: 7.31397274617e-7",
            "extra": "mean: 8.89740348425 usec\nrounds: 84382"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-compile]",
            "value": 42937.0770068,
            "unit": "iter/sec",
            "range": "stddev: 0.00000213653476947",
            "extra": "mean: 23.2898946485 usec\nrounds: 27525"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-eager]",
            "value": 67141.9518944,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011707373298",
            "extra": "mean: 14.893817826 usec\nrounds: 45331"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-compile]",
            "value": 35420.4815285,
            "unit": "iter/sec",
            "range": "stddev: 0.00000194146903256",
            "extra": "mean: 28.2322531159 usec\nrounds: 22968"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-eager]",
            "value": 36785.1497573,
            "unit": "iter/sec",
            "range": "stddev: 0.00000124639765186",
            "extra": "mean: 27.1848832096 usec\nrounds: 33190"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-compile]",
            "value": 20640.6842179,
            "unit": "iter/sec",
            "range": "stddev: 0.00000478571801759",
            "extra": "mean: 48.4480063473 usec\nrounds: 18282"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[eager]",
            "value": 38434.835937,
            "unit": "iter/sec",
            "range": "stddev: 0.00000193138289136",
            "extra": "mean: 26.0180634474 usec\nrounds: 29797"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[compile]",
            "value": 8396.40690584,
            "unit": "iter/sec",
            "range": "stddev: 0.00000795206415441",
            "extra": "mean: 119.098563375 usec\nrounds: 7303"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[eager]",
            "value": 32408.5826471,
            "unit": "iter/sec",
            "range": "stddev: 0.00000327885867711",
            "extra": "mean: 30.8560238777 usec\nrounds: 19643"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile]",
            "value": 8418.27954187,
            "unit": "iter/sec",
            "range": "stddev: 0.0000124683367282",
            "extra": "mean: 118.789117779 usec\nrounds: 5751"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile-overhead]",
            "value": 5302.27881049,
            "unit": "iter/sec",
            "range": "stddev: 0.0000112652780853",
            "extra": "mean: 188.59815482 usec\nrounds: 317"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[eager]",
            "value": 3797.14933028,
            "unit": "iter/sec",
            "range": "stddev: 0.00000816067380951",
            "extra": "mean: 263.355457744 usec\nrounds: 3428"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile]",
            "value": 3148.3149601,
            "unit": "iter/sec",
            "range": "stddev: 0.0000146891874888",
            "extra": "mean: 317.630228447 usec\nrounds: 2547"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile-overhead]",
            "value": 252.489376684,
            "unit": "iter/sec",
            "range": "stddev: 0.00177311975489",
            "extra": "mean: 3.96056267052 msec\nrounds: 1032"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[False]",
            "value": 6380.18823695,
            "unit": "iter/sec",
            "range": "stddev: 0.00000606144936263",
            "extra": "mean: 156.735187562 usec\nrounds: 5213"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[True]",
            "value": 6322.03811133,
            "unit": "iter/sec",
            "range": "stddev: 0.00000487937627494",
            "extra": "mean: 158.17683829 usec\nrounds: 5419"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[eager]",
            "value": 662.413534191,
            "unit": "iter/sec",
            "range": "stddev: 0.0000642025944796",
            "extra": "mean: 1.50963099089 msec\nrounds: 654"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile]",
            "value": 753.453402902,
            "unit": "iter/sec",
            "range": "stddev: 0.0000528464751018",
            "extra": "mean: 1.32722208984 msec\nrounds: 678"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile-overhead]",
            "value": 1067.47772973,
            "unit": "iter/sec",
            "range": "stddev: 0.0000758422055188",
            "extra": "mean: 936.787693222 usec\nrounds: 88"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[eager]",
            "value": 10335.3221875,
            "unit": "iter/sec",
            "range": "stddev: 0.00000472150971311",
            "extra": "mean: 96.7555710273 usec\nrounds: 8512"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile]",
            "value": 8120.86871657,
            "unit": "iter/sec",
            "range": "stddev: 0.00000413728444334",
            "extra": "mean: 123.139535302 usec\nrounds: 6867"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile-overhead]",
            "value": 4903.59960983,
            "unit": "iter/sec",
            "range": "stddev: 0.00000840743434093",
            "extra": "mean: 203.931821431 usec\nrounds: 3348"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[eager]",
            "value": 2533.86461083,
            "unit": "iter/sec",
            "range": "stddev: 0.00000917520784724",
            "extra": "mean: 394.654077304 usec\nrounds: 2431"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile]",
            "value": 2918.51121184,
            "unit": "iter/sec",
            "range": "stddev: 0.0000178297079752",
            "extra": "mean: 342.640451729 usec\nrounds: 2734"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile-overhead]",
            "value": 3318.53262642,
            "unit": "iter/sec",
            "range": "stddev: 0.00000733547593724",
            "extra": "mean: 301.338004647 usec\nrounds: 2367"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-eager]",
            "value": 1245.57100187,
            "unit": "iter/sec",
            "range": "stddev: 0.0000123159443998",
            "extra": "mean: 802.844637919 usec\nrounds: 1215"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile]",
            "value": 1403.83776984,
            "unit": "iter/sec",
            "range": "stddev: 0.0000562514787111",
            "extra": "mean: 712.333021299 usec\nrounds: 1371"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile-overhead]",
            "value": 2013.37226437,
            "unit": "iter/sec",
            "range": "stddev: 0.0000599561838967",
            "extra": "mean: 496.679137631 usec\nrounds: 1351"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-eager]",
            "value": 753.266965078,
            "unit": "iter/sec",
            "range": "stddev: 0.000157726230148",
            "extra": "mean: 1.3275505848 msec\nrounds: 768"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile]",
            "value": 1370.09147055,
            "unit": "iter/sec",
            "range": "stddev: 0.0000178107223512",
            "extra": "mean: 729.878275643 usec\nrounds: 1302"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile-overhead]",
            "value": 2031.57294393,
            "unit": "iter/sec",
            "range": "stddev: 0.0000139346877522",
            "extra": "mean: 492.229433843 usec\nrounds: 1268"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-eager]",
            "value": 1213.88525552,
            "unit": "iter/sec",
            "range": "stddev: 0.000027351856397",
            "extra": "mean: 823.801092778 usec\nrounds: 1208"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile]",
            "value": 1381.33988391,
            "unit": "iter/sec",
            "range": "stddev: 0.00001933490496",
            "extra": "mean: 723.934790883 usec\nrounds: 1392"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile-overhead]",
            "value": 2179.15823831,
            "unit": "iter/sec",
            "range": "stddev: 0.0000379130236373",
            "extra": "mean: 458.892788243 usec\nrounds: 1119"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-eager]",
            "value": 758.701867464,
            "unit": "iter/sec",
            "range": "stddev: 0.0000193824300634",
            "extra": "mean: 1.31804077844 msec\nrounds: 745"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile]",
            "value": 1203.0203843,
            "unit": "iter/sec",
            "range": "stddev: 0.000057734421861",
            "extra": "mean: 831.241110331 usec\nrounds: 1189"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile-overhead]",
            "value": 1349.0269664,
            "unit": "iter/sec",
            "range": "stddev: 0.000153289056098",
            "extra": "mean: 741.275026301 usec\nrounds: 952"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[eager]",
            "value": 346.986785797,
            "unit": "iter/sec",
            "range": "stddev: 0.0000816264829247",
            "extra": "mean: 2.8819541289 msec\nrounds: 342"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile]",
            "value": 1287.23878464,
            "unit": "iter/sec",
            "range": "stddev: 0.0000268010137522",
            "extra": "mean: 776.856642241 usec\nrounds: 1026"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile-overhead]",
            "value": 1970.52702457,
            "unit": "iter/sec",
            "range": "stddev: 0.0000315060919875",
            "extra": "mean: 507.478449943 usec\nrounds: 129"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1043.60049951,
            "unit": "iter/sec",
            "range": "stddev: 0.0000738344744514",
            "extra": "mean: 958.221082179 usec\nrounds: 961"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1059.43509887,
            "unit": "iter/sec",
            "range": "stddev: 0.000046657413249",
            "extra": "mean: 943.899254482 usec\nrounds: 1025"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1149.17081892,
            "unit": "iter/sec",
            "range": "stddev: 0.00004538915528",
            "extra": "mean: 870.192649808 usec\nrounds: 1151"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1164.39556302,
            "unit": "iter/sec",
            "range": "stddev: 0.0000323905971227",
            "extra": "mean: 858.814677555 usec\nrounds: 1154"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-True]",
            "value": 37.961636236,
            "unit": "iter/sec",
            "range": "stddev: 0.00038502407832",
            "extra": "mean: 26.3423840264 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-False]",
            "value": 38.1320534528,
            "unit": "iter/sec",
            "range": "stddev: 0.000331108848929",
            "extra": "mean: 26.2246564098 msec\nrounds: 39"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-True]",
            "value": 38.3552902693,
            "unit": "iter/sec",
            "range": "stddev: 0.000265847222445",
            "extra": "mean: 26.0720227374 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-False]",
            "value": 37.7961937255,
            "unit": "iter/sec",
            "range": "stddev: 0.00085783093691",
            "extra": "mean: 26.4576906146 msec\nrounds: 39"
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
          "id": "b444822d2a22a30288107ba77765e4783ab23bf6",
          "message": "[Test] Fix the failures of the first nightly Windows and Linux ARM run, which came from the test setup (#3071)",
          "timestamp": "2026-10-20T11:02:25-04:00",
          "tree_id": "76d62fca0a691ad85a3185c9c16bbd19d81b371f",
          "url": "https://github.com/pytorch/tensordict/commit/b444822d2a22a30288107ba77765e4783ab23bf6"
        },
        "date": 1792509888625,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-2]",
            "value": 4121.04990741,
            "unit": "iter/sec",
            "range": "stddev: 0.0000114035900414",
            "extra": "mean: 242.656609958 usec\nrounds: 141"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-8]",
            "value": 3301.74318047,
            "unit": "iter/sec",
            "range": "stddev: 0.000011456512564",
            "extra": "mean: 302.870315872 usec\nrounds: 2900"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-32]",
            "value": 1914.72566218,
            "unit": "iter/sec",
            "range": "stddev: 0.0000142905186132",
            "extra": "mean: 522.268030221 usec\nrounds: 1720"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-2]",
            "value": 4149.73792548,
            "unit": "iter/sec",
            "range": "stddev: 0.0000106436897725",
            "extra": "mean: 240.979073367 usec\nrounds: 3639"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-8]",
            "value": 3318.19542642,
            "unit": "iter/sec",
            "range": "stddev: 0.0000794112121128",
            "extra": "mean: 301.368627067 usec\nrounds: 2963"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-32]",
            "value": 1739.14521458,
            "unit": "iter/sec",
            "range": "stddev: 0.0000142961335138",
            "extra": "mean: 574.995113472 usec\nrounds: 1595"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-2]",
            "value": 2178.28128677,
            "unit": "iter/sec",
            "range": "stddev: 0.000016936890445",
            "extra": "mean: 459.077533318 usec\nrounds: 75"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-8]",
            "value": 583.021083972,
            "unit": "iter/sec",
            "range": "stddev: 0.000104627806526",
            "extra": "mean: 1.71520383652 msec\nrounds: 575"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-32]",
            "value": 151.416199734,
            "unit": "iter/sec",
            "range": "stddev: 0.0000700317803877",
            "extra": "mean: 6.60431315644 msec\nrounds: 147"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-2]",
            "value": 2189.52550174,
            "unit": "iter/sec",
            "range": "stddev: 0.0000133280263308",
            "extra": "mean: 456.719960195 usec\nrounds: 2085"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-8]",
            "value": 583.326107618,
            "unit": "iter/sec",
            "range": "stddev: 0.0000261292911695",
            "extra": "mean: 1.7143069493 msec\nrounds: 572"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-32]",
            "value": 148.810092131,
            "unit": "iter/sec",
            "range": "stddev: 0.0000683139328158",
            "extra": "mean: 6.7199743356 msec\nrounds: 146"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-2]",
            "value": 2972.16418556,
            "unit": "iter/sec",
            "range": "stddev: 0.00000923548689233",
            "extra": "mean: 336.455167873 usec\nrounds: 1930"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-8]",
            "value": 798.29350263,
            "unit": "iter/sec",
            "range": "stddev: 0.0000163980649322",
            "extra": "mean: 1.25267210206 msec\nrounds: 784"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-32]",
            "value": 203.620896596,
            "unit": "iter/sec",
            "range": "stddev: 0.0000370442401506",
            "extra": "mean: 4.91108730351 msec\nrounds: 201"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-2]",
            "value": 2940.82559453,
            "unit": "iter/sec",
            "range": "stddev: 0.0000106232354429",
            "extra": "mean: 340.040566111 usec\nrounds: 2798"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-8]",
            "value": 794.205336674,
            "unit": "iter/sec",
            "range": "stddev: 0.0000216241868598",
            "extra": "mean: 1.25912022222 msec\nrounds: 792"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-32]",
            "value": 202.371157278,
            "unit": "iter/sec",
            "range": "stddev: 0.0000496132763708",
            "extra": "mean: 4.9414156318 msec\nrounds: 201"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-2]",
            "value": 3453.44501501,
            "unit": "iter/sec",
            "range": "stddev: 0.0000100591412818",
            "extra": "mean: 289.56592494 usec\nrounds: 2798"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-8]",
            "value": 929.473883141,
            "unit": "iter/sec",
            "range": "stddev: 0.0000223770023965",
            "extra": "mean: 1.07587745943 msec\nrounds: 912"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-32]",
            "value": 239.839019728,
            "unit": "iter/sec",
            "range": "stddev: 0.0000919907331926",
            "extra": "mean: 4.16946333893 msec\nrounds: 239"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-2]",
            "value": 3447.46231046,
            "unit": "iter/sec",
            "range": "stddev: 0.0000104834326216",
            "extra": "mean: 290.068435836 usec\nrounds: 3187"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-8]",
            "value": 926.472172428,
            "unit": "iter/sec",
            "range": "stddev: 0.0000202273668594",
            "extra": "mean: 1.07936323374 msec\nrounds: 907"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-32]",
            "value": 235.119912373,
            "unit": "iter/sec",
            "range": "stddev: 0.0000419945963036",
            "extra": "mean: 4.25314891413 msec\nrounds: 233"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-128]",
            "value": 184.733746488,
            "unit": "iter/sec",
            "range": "stddev: 0.0000545652750303",
            "extra": "mean: 5.4131961215 msec\nrounds: 181"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-256]",
            "value": 97.356670837,
            "unit": "iter/sec",
            "range": "stddev: 0.00010711396081",
            "extra": "mean: 10.2715098144 msec\nrounds: 97"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-128]",
            "value": 178.811575784,
            "unit": "iter/sec",
            "range": "stddev: 0.0000488586199794",
            "extra": "mean: 5.5924790977 msec\nrounds: 174"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-256]",
            "value": 92.8010162012,
            "unit": "iter/sec",
            "range": "stddev: 0.0000827891180501",
            "extra": "mean: 10.7757440698 msec\nrounds: 86"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-128]",
            "value": 12.4506167953,
            "unit": "iter/sec",
            "range": "stddev: 0.000584344754144",
            "extra": "mean: 80.3173060771 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-256]",
            "value": 6.13764985901,
            "unit": "iter/sec",
            "range": "stddev: 0.00128960773526",
            "extra": "mean: 162.928812 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-128]",
            "value": 12.2165245573,
            "unit": "iter/sec",
            "range": "stddev: 0.000601024353549",
            "extra": "mean: 81.856341 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-256]",
            "value": 6.08361762607,
            "unit": "iter/sec",
            "range": "stddev: 0.000317529025395",
            "extra": "mean: 164.375879857 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-128]",
            "value": 22.8016973548,
            "unit": "iter/sec",
            "range": "stddev: 0.00009264881249",
            "extra": "mean: 43.8563842174 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-256]",
            "value": 11.4455305874,
            "unit": "iter/sec",
            "range": "stddev: 0.000153727755558",
            "extra": "mean: 87.3703488332 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-128]",
            "value": 22.7247209791,
            "unit": "iter/sec",
            "range": "stddev: 0.000275842082649",
            "extra": "mean: 44.0049407393 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-256]",
            "value": 11.3899954881,
            "unit": "iter/sec",
            "range": "stddev: 0.000340195380873",
            "extra": "mean: 87.7963473334 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-128]",
            "value": 26.7166595925,
            "unit": "iter/sec",
            "range": "stddev: 0.00026178602515",
            "extra": "mean: 37.4298290001 msec\nrounds: 27"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-256]",
            "value": 13.2290199409,
            "unit": "iter/sec",
            "range": "stddev: 0.000272870419025",
            "extra": "mean: 75.5913895715 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-128]",
            "value": 26.7002645391,
            "unit": "iter/sec",
            "range": "stddev: 0.000324766915263",
            "extra": "mean: 37.4528124444 msec\nrounds: 27"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-256]",
            "value": 13.2424181296,
            "unit": "iter/sec",
            "range": "stddev: 0.00147413495981",
            "extra": "mean: 75.5149090001 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_mask_many_leaves[cuda]",
            "value": 5989.70800712,
            "unit": "iter/sec",
            "range": "stddev: 0.0000162580344393",
            "extra": "mean: 166.953046594 usec\nrounds: 43"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-None]",
            "value": 6503.44235386,
            "unit": "iter/sec",
            "range": "stddev: 0.00000992382143963",
            "extra": "mean: 153.764721141 usec\nrounds: 1144"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-cpu]",
            "value": 6576.60800764,
            "unit": "iter/sec",
            "range": "stddev: 0.00000828915420415",
            "extra": "mean: 152.054067817 usec\nrounds: 5028"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to",
            "value": 22128.4429864,
            "unit": "iter/sec",
            "range": "stddev: 0.0000046198849564",
            "extra": "mean: 45.1907077517 usec\nrounds: 11733"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_nonblocking",
            "value": 22456.5126851,
            "unit": "iter/sec",
            "range": "stddev: 0.0000055892501616",
            "extra": "mean: 44.5305116615 usec\nrounds: 12352"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[add]",
            "value": 10149.6738577,
            "unit": "iter/sec",
            "range": "stddev: 0.00000890534052313",
            "extra": "mean: 98.5253333277 usec\nrounds: 189"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[sub]",
            "value": 10490.7777851,
            "unit": "iter/sec",
            "range": "stddev: 0.00000706615165701",
            "extra": "mean: 95.3218169789 usec\nrounds: 7649"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_blocking",
            "value": 39.1240388063,
            "unit": "iter/sec",
            "range": "stddev: 0.00103470714171",
            "extra": "mean: 25.5597333637 msec\nrounds: 55"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_global_sync",
            "value": 90.5021045736,
            "unit": "iter/sec",
            "range": "stddev: 0.0141439585657",
            "extra": "mean: 11.049466802 msec\nrounds: 101"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_event_sync",
            "value": 93.9241519465,
            "unit": "iter/sec",
            "range": "stddev: 0.000185016268613",
            "extra": "mean: 10.646888785 msec\nrounds: 93"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_default",
            "value": 93.5058993243,
            "unit": "iter/sec",
            "range": "stddev: 0.000133940809606",
            "extra": "mean: 10.6945124022 msec\nrounds: 92"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_cpu_to_cpu_default",
            "value": 3901.01818198,
            "unit": "iter/sec",
            "range": "stddev: 0.00000693822327317",
            "extra": "mean: 256.343332267 usec\nrounds: 3774"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[False-None]",
            "value": 291.82467287,
            "unit": "iter/sec",
            "range": "stddev: 0.0000367620668752",
            "extra": "mean: 3.42671505519 msec\nrounds: 290"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[default-None]",
            "value": 441.600350818,
            "unit": "iter/sec",
            "range": "stddev: 0.000211739174166",
            "extra": "mean: 2.26449095466 msec\nrounds: 441"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[reduce-overhead-None]",
            "value": 446.716141199,
            "unit": "iter/sec",
            "range": "stddev: 0.000243849935144",
            "extra": "mean: 2.23855801878 msec\nrounds: 426"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[False-False-None]",
            "value": 484.134600407,
            "unit": "iter/sec",
            "range": "stddev: 0.00013524638862",
            "extra": "mean: 2.06554127542 msec\nrounds: 443"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-False-None]",
            "value": 643.03100749,
            "unit": "iter/sec",
            "range": "stddev: 0.0000444134049486",
            "extra": "mean: 1.55513495983 msec\nrounds: 622"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[within-False-None]",
            "value": 197.024985882,
            "unit": "iter/sec",
            "range": "stddev: 0.0000361718064474",
            "extra": "mean: 5.07549839694 msec\nrounds: 194"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-default-None]",
            "value": 115.944015005,
            "unit": "iter/sec",
            "range": "stddev: 0.0000883915718401",
            "extra": "mean: 8.6248522613 msec\nrounds: 111"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[False-False-None]",
            "value": 138.839751946,
            "unit": "iter/sec",
            "range": "stddev: 0.000051227389736",
            "extra": "mean: 7.2025481606 msec\nrounds: 137"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[True-False-None]",
            "value": 264.608533623,
            "unit": "iter/sec",
            "range": "stddev: 0.000420015226415",
            "extra": "mean: 3.77916761152 msec\nrounds: 260"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[within-False-None]",
            "value": 113.807438511,
            "unit": "iter/sec",
            "range": "stddev: 0.0000336508123532",
            "extra": "mean: 8.78677187609 msec\nrounds: 113"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 15.691360135,
            "unit": "iter/sec",
            "range": "stddev: 0.000394730477655",
            "extra": "mean: 63.7293384 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_archive",
            "value": 4.42394274941,
            "unit": "iter/sec",
            "range": "stddev: 0.00240925870085",
            "extra": "mean: 226.0427082 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_zarr",
            "value": 0.655378329565,
            "unit": "iter/sec",
            "range": "stddev: 0.0814748629291",
            "extra": "mean: 1.5258362306 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 41.5397281803,
            "unit": "iter/sec",
            "range": "stddev: 0.000539950409376",
            "extra": "mean: 24.0733400002 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 7.97619020442,
            "unit": "iter/sec",
            "range": "stddev: 0.12409547932",
            "extra": "mean: 125.373138601 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 14.5533860245,
            "unit": "iter/sec",
            "range": "stddev: 0.000720880356506",
            "extra": "mean: 68.7125317998 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 42.8431029275,
            "unit": "iter/sec",
            "range": "stddev: 0.000379583898902",
            "extra": "mean: 23.3409798 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-compile]",
            "value": 4529.13213608,
            "unit": "iter/sec",
            "range": "stddev: 0.0000101213024844",
            "extra": "mean: 220.792851689 usec\nrounds: 1679"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-eager]",
            "value": 8021.8866847,
            "unit": "iter/sec",
            "range": "stddev: 0.00000635501036279",
            "extra": "mean: 124.658953598 usec\nrounds: 6896"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-compile]",
            "value": 6304.75435977,
            "unit": "iter/sec",
            "range": "stddev: 0.00000869860968791",
            "extra": "mean: 158.610461714 usec\nrounds: 2965"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-eager]",
            "value": 5509.77135876,
            "unit": "iter/sec",
            "range": "stddev: 0.0000173206357287",
            "extra": "mean: 181.495734557 usec\nrounds: 5150"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-compile]",
            "value": 21413.6152131,
            "unit": "iter/sec",
            "range": "stddev: 0.00000585714222692",
            "extra": "mean: 46.6992607295 usec\nrounds: 15472"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-eager]",
            "value": 31272.5435464,
            "unit": "iter/sec",
            "range": "stddev: 0.00000180617787683",
            "extra": "mean: 31.9769320495 usec\nrounds: 25680"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-compile]",
            "value": 45815.9982335,
            "unit": "iter/sec",
            "range": "stddev: 0.00000537421644283",
            "extra": "mean: 21.826437021 usec\nrounds: 32582"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-eager]",
            "value": 15557.8073353,
            "unit": "iter/sec",
            "range": "stddev: 0.000011205215022",
            "extra": "mean: 64.2764098082 usec\nrounds: 14719"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-compile]",
            "value": 3613.48991234,
            "unit": "iter/sec",
            "range": "stddev: 0.00000925957137813",
            "extra": "mean: 276.740775334 usec\nrounds: 1273"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-eager]",
            "value": 5207.25456176,
            "unit": "iter/sec",
            "range": "stddev: 0.00000710163987745",
            "extra": "mean: 192.039776074 usec\nrounds: 4921"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-compile]",
            "value": 5689.877083,
            "unit": "iter/sec",
            "range": "stddev: 0.00000727796308081",
            "extra": "mean: 175.750721046 usec\nrounds: 3273"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-eager]",
            "value": 15515.8105834,
            "unit": "iter/sec",
            "range": "stddev: 0.00000376718506593",
            "extra": "mean: 64.4503872114 usec\nrounds: 13029"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-compile]",
            "value": 4001.8243593,
            "unit": "iter/sec",
            "range": "stddev: 0.00000865604291323",
            "extra": "mean: 249.886029525 usec\nrounds: 1355"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-eager]",
            "value": 1856.14693701,
            "unit": "iter/sec",
            "range": "stddev: 0.0000221791597212",
            "extra": "mean: 538.750451305 usec\nrounds: 1828"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-eager]",
            "value": 4273.30329431,
            "unit": "iter/sec",
            "range": "stddev: 0.00000900440666218",
            "extra": "mean: 234.011005334 usec\nrounds: 3935"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-compile]",
            "value": 3604.8316676,
            "unit": "iter/sec",
            "range": "stddev: 0.00000984349464813",
            "extra": "mean: 277.405463613 usec\nrounds: 1333"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-eager]",
            "value": 12610.3340276,
            "unit": "iter/sec",
            "range": "stddev: 0.00000640244430773",
            "extra": "mean: 79.3000405709 usec\nrounds: 10994"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-compile]",
            "value": 5677.53405521,
            "unit": "iter/sec",
            "range": "stddev: 0.00000902763016184",
            "extra": "mean: 176.132805242 usec\nrounds: 3245"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-eager]",
            "value": 2098.42695425,
            "unit": "iter/sec",
            "range": "stddev: 0.0000191937574997",
            "extra": "mean: 476.5474433 usec\nrounds: 2055"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-compile]",
            "value": 4021.78161138,
            "unit": "iter/sec",
            "range": "stddev: 0.00000953977693931",
            "extra": "mean: 248.646022244 usec\nrounds: 1349"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-compile]",
            "value": 32854.9478881,
            "unit": "iter/sec",
            "range": "stddev: 0.00000275715113485",
            "extra": "mean: 30.4368158916 usec\nrounds: 21123"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-eager]",
            "value": 53429.4794499,
            "unit": "iter/sec",
            "range": "stddev: 0.00000241322733647",
            "extra": "mean: 18.7162594563 usec\nrounds: 46081"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-compile]",
            "value": 45228.5068049,
            "unit": "iter/sec",
            "range": "stddev: 0.00000209686617311",
            "extra": "mean: 22.1099494687 usec\nrounds: 28042"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-eager]",
            "value": 19525.3637192,
            "unit": "iter/sec",
            "range": "stddev: 0.0000114476503637",
            "extra": "mean: 51.2154351837 usec\nrounds: 16747"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-compile]",
            "value": 3866.76569033,
            "unit": "iter/sec",
            "range": "stddev: 0.000225484552866",
            "extra": "mean: 258.614066661 usec\nrounds: 90"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-eager]",
            "value": 293.487189555,
            "unit": "iter/sec",
            "range": "stddev: 0.0000674355766506",
            "extra": "mean: 3.40730374473 msec\nrounds: 286"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-compile]",
            "value": 4967.47240775,
            "unit": "iter/sec",
            "range": "stddev: 0.000217865380309",
            "extra": "mean: 201.30962347 usec\nrounds: 85"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-eager]",
            "value": 341.190152329,
            "unit": "iter/sec",
            "range": "stddev: 0.0000480705888872",
            "extra": "mean: 2.93091694814 msec\nrounds: 328"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-compile]",
            "value": 6066.15479283,
            "unit": "iter/sec",
            "range": "stddev: 0.00000649125389592",
            "extra": "mean: 164.849073944 usec\nrounds: 3827"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-eager]",
            "value": 17339.1537421,
            "unit": "iter/sec",
            "range": "stddev: 0.0000127396403689",
            "extra": "mean: 57.672941533 usec\nrounds: 14196"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-compile]",
            "value": 7413.77828039,
            "unit": "iter/sec",
            "range": "stddev: 0.00000592999875218",
            "extra": "mean: 134.883990616 usec\nrounds: 4373"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-eager]",
            "value": 19820.0284403,
            "unit": "iter/sec",
            "range": "stddev: 0.0000116669494139",
            "extra": "mean: 50.454014383 usec\nrounds: 16205"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-compile]",
            "value": 7362.30554909,
            "unit": "iter/sec",
            "range": "stddev: 0.00000657329520056",
            "extra": "mean: 135.827016867 usec\nrounds: 6761"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-eager]",
            "value": 19990.7977602,
            "unit": "iter/sec",
            "range": "stddev: 0.0000120270522418",
            "extra": "mean: 50.0230161895 usec\nrounds: 17295"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-compile]",
            "value": 11355.2088594,
            "unit": "iter/sec",
            "range": "stddev: 0.0000118478613833",
            "extra": "mean: 88.0653110292 usec\nrounds: 9253"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-eager]",
            "value": 44318.9827818,
            "unit": "iter/sec",
            "range": "stddev: 0.0000111651710878",
            "extra": "mean: 22.5636947699 usec\nrounds: 32153"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-compile]",
            "value": 15275.1481189,
            "unit": "iter/sec",
            "range": "stddev: 0.00000425378756445",
            "extra": "mean: 65.4658136349 usec\nrounds: 12465"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-eager]",
            "value": 44284.199516,
            "unit": "iter/sec",
            "range": "stddev: 0.0000109795035737",
            "extra": "mean: 22.5814175469 usec\nrounds: 34221"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-compile]",
            "value": 15234.9414033,
            "unit": "iter/sec",
            "range": "stddev: 0.00000527906189176",
            "extra": "mean: 65.6385852446 usec\nrounds: 13268"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-eager]",
            "value": 45142.0548308,
            "unit": "iter/sec",
            "range": "stddev: 0.0000109281407329",
            "extra": "mean: 22.152292441 usec\nrounds: 35498"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-compile]",
            "value": 11183.7722815,
            "unit": "iter/sec",
            "range": "stddev: 0.0000056791091938",
            "extra": "mean: 89.4152683758 usec\nrounds: 10247"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-eager]",
            "value": 44783.8026982,
            "unit": "iter/sec",
            "range": "stddev: 0.0000105253195429",
            "extra": "mean: 22.3295017339 usec\nrounds: 36322"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-compile]",
            "value": 15044.6273169,
            "unit": "iter/sec",
            "range": "stddev: 0.00000453076757205",
            "extra": "mean: 66.4689113885 usec\nrounds: 13396"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-eager]",
            "value": 44029.1097838,
            "unit": "iter/sec",
            "range": "stddev: 0.0000106689894864",
            "extra": "mean: 22.712246623 usec\nrounds: 34964"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-compile]",
            "value": 15091.6537909,
            "unit": "iter/sec",
            "range": "stddev: 0.0000040940518649",
            "extra": "mean: 66.2617903813 usec\nrounds: 13191"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-eager]",
            "value": 43806.0424333,
            "unit": "iter/sec",
            "range": "stddev: 0.000010883973025",
            "extra": "mean: 22.827901003 usec\nrounds: 35738"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-eager]",
            "value": 30558.0532424,
            "unit": "iter/sec",
            "range": "stddev: 0.00000316909583884",
            "extra": "mean: 32.7245977375 usec\nrounds: 21762"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-compile]",
            "value": 6566.99702031,
            "unit": "iter/sec",
            "range": "stddev: 0.00000980079754979",
            "extra": "mean: 152.27660328 usec\nrounds: 3660"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-eager]",
            "value": 2493.6532097,
            "unit": "iter/sec",
            "range": "stddev: 0.0000146242628482",
            "extra": "mean: 401.018071041 usec\nrounds: 2351"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-compile]",
            "value": 6239.49536307,
            "unit": "iter/sec",
            "range": "stddev: 0.00000614220817077",
            "extra": "mean: 160.269371449 usec\nrounds: 3481"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[eager]",
            "value": 5832.15890476,
            "unit": "iter/sec",
            "range": "stddev: 0.00000503666322308",
            "extra": "mean: 171.463092198 usec\nrounds: 5629"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[compile]",
            "value": 5626.32013674,
            "unit": "iter/sec",
            "range": "stddev: 0.00000651691311844",
            "extra": "mean: 177.736064727 usec\nrounds: 3661"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[eager]",
            "value": 5512.38976754,
            "unit": "iter/sec",
            "range": "stddev: 0.00000751918677868",
            "extra": "mean: 181.409523305 usec\nrounds: 5257"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[compile]",
            "value": 4812.80722693,
            "unit": "iter/sec",
            "range": "stddev: 0.000010077673783",
            "extra": "mean: 207.778943317 usec\nrounds: 2964"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-eager]",
            "value": 107425.276435,
            "unit": "iter/sec",
            "range": "stddev: 0.00000119890851239",
            "extra": "mean: 9.30879615286 usec\nrounds: 83676"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-compile]",
            "value": 38999.8024023,
            "unit": "iter/sec",
            "range": "stddev: 0.00000365068994115",
            "extra": "mean: 25.6411555547 usec\nrounds: 24461"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-eager]",
            "value": 65741.7021501,
            "unit": "iter/sec",
            "range": "stddev: 0.00000147513687496",
            "extra": "mean: 15.2110451554 usec\nrounds: 54612"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-compile]",
            "value": 33001.3344072,
            "unit": "iter/sec",
            "range": "stddev: 0.00000287282495654",
            "extra": "mean: 30.3018050016 usec\nrounds: 20605"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-eager]",
            "value": 36806.2363145,
            "unit": "iter/sec",
            "range": "stddev: 0.00000177300198948",
            "extra": "mean: 27.1693087947 usec\nrounds: 32808"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-compile]",
            "value": 23286.2978401,
            "unit": "iter/sec",
            "range": "stddev: 0.00000334887316273",
            "extra": "mean: 42.9437090801 usec\nrounds: 16372"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[eager]",
            "value": 36016.4225946,
            "unit": "iter/sec",
            "range": "stddev: 0.00000247641867058",
            "extra": "mean: 27.7651118007 usec\nrounds: 26931"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[compile]",
            "value": 8258.85126357,
            "unit": "iter/sec",
            "range": "stddev: 0.00000648188735011",
            "extra": "mean: 121.082214473 usec\nrounds: 6826"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[eager]",
            "value": 29497.2504236,
            "unit": "iter/sec",
            "range": "stddev: 0.00000334460691706",
            "extra": "mean: 33.901464904 usec\nrounds: 20316"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile]",
            "value": 8803.42374141,
            "unit": "iter/sec",
            "range": "stddev: 0.0000242610854726",
            "extra": "mean: 113.592169294 usec\nrounds: 5523"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile-overhead]",
            "value": 5049.73453465,
            "unit": "iter/sec",
            "range": "stddev: 0.0000131173598419",
            "extra": "mean: 198.030211913 usec\nrounds: 302"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[eager]",
            "value": 3499.27230662,
            "unit": "iter/sec",
            "range": "stddev: 0.0000111243851182",
            "extra": "mean: 285.773701609 usec\nrounds: 3167"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile]",
            "value": 3040.04886896,
            "unit": "iter/sec",
            "range": "stddev: 0.00000942223954186",
            "extra": "mean: 328.942080573 usec\nrounds: 2383"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile-overhead]",
            "value": 262.080977927,
            "unit": "iter/sec",
            "range": "stddev: 0.00168107914178",
            "extra": "mean: 3.81561457802 msec\nrounds: 955"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[False]",
            "value": 5902.37009831,
            "unit": "iter/sec",
            "range": "stddev: 0.00000775062799163",
            "extra": "mean: 169.423466056 usec\nrounds: 4965"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[True]",
            "value": 5888.18598966,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113092564911",
            "extra": "mean: 169.831591895 usec\nrounds: 5077"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[eager]",
            "value": 644.083779616,
            "unit": "iter/sec",
            "range": "stddev: 0.0000320220519699",
            "extra": "mean: 1.55259305023 msec\nrounds: 617"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile]",
            "value": 743.235084171,
            "unit": "iter/sec",
            "range": "stddev: 0.000045504267379",
            "extra": "mean: 1.34546931556 msec\nrounds: 656"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile-overhead]",
            "value": 1093.24343956,
            "unit": "iter/sec",
            "range": "stddev: 0.0000819132427974",
            "extra": "mean: 914.709353665 usec\nrounds: 82"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[eager]",
            "value": 9982.36983821,
            "unit": "iter/sec",
            "range": "stddev: 0.00000563015465245",
            "extra": "mean: 100.176612989 usec\nrounds: 8129"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile]",
            "value": 7741.10212683,
            "unit": "iter/sec",
            "range": "stddev: 0.00000617116737912",
            "extra": "mean: 129.180571915 usec\nrounds: 6487"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile-overhead]",
            "value": 4838.8578785,
            "unit": "iter/sec",
            "range": "stddev: 0.00000876517253357",
            "extra": "mean: 206.660337028 usec\nrounds: 3255"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[eager]",
            "value": 2408.49197695,
            "unit": "iter/sec",
            "range": "stddev: 0.0000133818120521",
            "extra": "mean: 415.197563277 usec\nrounds: 2331"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile]",
            "value": 2841.93936721,
            "unit": "iter/sec",
            "range": "stddev: 0.0000108951829603",
            "extra": "mean: 351.872390924 usec\nrounds: 2599"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile-overhead]",
            "value": 3230.65429241,
            "unit": "iter/sec",
            "range": "stddev: 0.0000107807106747",
            "extra": "mean: 309.534821584 usec\nrounds: 2270"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-eager]",
            "value": 1153.88192913,
            "unit": "iter/sec",
            "range": "stddev: 0.0000178340436654",
            "extra": "mean: 866.639796287 usec\nrounds: 1134"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile]",
            "value": 1375.94446576,
            "unit": "iter/sec",
            "range": "stddev: 0.0000177894007944",
            "extra": "mean: 726.773518032 usec\nrounds: 1303"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile-overhead]",
            "value": 2105.70398433,
            "unit": "iter/sec",
            "range": "stddev: 0.0000165452067035",
            "extra": "mean: 474.900559358 usec\nrounds: 1314"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-eager]",
            "value": 749.751528629,
            "unit": "iter/sec",
            "range": "stddev: 0.0000219465800761",
            "extra": "mean: 1.3337752066 msec\nrounds: 726"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile]",
            "value": 1289.25016633,
            "unit": "iter/sec",
            "range": "stddev: 0.000022438398202",
            "extra": "mean: 775.644654631 usec\nrounds: 1245"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile-overhead]",
            "value": 1987.11485319,
            "unit": "iter/sec",
            "range": "stddev: 0.0000143303992343",
            "extra": "mean: 503.242174651 usec\nrounds: 1231"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-eager]",
            "value": 1145.83796459,
            "unit": "iter/sec",
            "range": "stddev: 0.0000193668488847",
            "extra": "mean: 872.723745333 usec\nrounds: 1123"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile]",
            "value": 1351.89234392,
            "unit": "iter/sec",
            "range": "stddev: 0.0000187288040299",
            "extra": "mean: 739.703871019 usec\nrounds: 1295"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile-overhead]",
            "value": 2206.99204276,
            "unit": "iter/sec",
            "range": "stddev: 0.0000104483602783",
            "extra": "mean: 453.105394412 usec\nrounds: 1108"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-eager]",
            "value": 722.707410561,
            "unit": "iter/sec",
            "range": "stddev: 0.0000235529973989",
            "extra": "mean: 1.38368582553 msec\nrounds: 705"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile]",
            "value": 1189.97769412,
            "unit": "iter/sec",
            "range": "stddev: 0.0000181676122059",
            "extra": "mean: 840.351886378 usec\nrounds: 1144"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile-overhead]",
            "value": 1314.15380201,
            "unit": "iter/sec",
            "range": "stddev: 0.0000816431824264",
            "extra": "mean: 760.945939868 usec\nrounds: 998"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[eager]",
            "value": 335.382114179,
            "unit": "iter/sec",
            "range": "stddev: 0.0000456625873228",
            "extra": "mean: 2.98167361265 msec\nrounds: 333"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile]",
            "value": 1270.62338732,
            "unit": "iter/sec",
            "range": "stddev: 0.0000157599684094",
            "extra": "mean: 787.015263514 usec\nrounds: 998"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile-overhead]",
            "value": 1927.59795921,
            "unit": "iter/sec",
            "range": "stddev: 0.0000369435900054",
            "extra": "mean: 518.780379084 usec\nrounds: 124"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1014.14080146,
            "unit": "iter/sec",
            "range": "stddev: 0.0000398878512562",
            "extra": "mean: 986.056372606 usec\nrounds: 891"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1017.42244787,
            "unit": "iter/sec",
            "range": "stddev: 0.0000384914525503",
            "extra": "mean: 982.875895945 usec\nrounds: 1009"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1105.25922798,
            "unit": "iter/sec",
            "range": "stddev: 0.0000385704499344",
            "extra": "mean: 904.765121778 usec\nrounds: 1043"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1105.68834896,
            "unit": "iter/sec",
            "range": "stddev: 0.0000377351631763",
            "extra": "mean: 904.413979709 usec\nrounds: 1084"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-True]",
            "value": 36.4374507257,
            "unit": "iter/sec",
            "range": "stddev: 0.000254332408757",
            "extra": "mean: 27.4442909722 msec\nrounds: 36"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-False]",
            "value": 36.6806513898,
            "unit": "iter/sec",
            "range": "stddev: 0.000222002149253",
            "extra": "mean: 27.2623293784 msec\nrounds: 37"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-True]",
            "value": 36.872634509,
            "unit": "iter/sec",
            "range": "stddev: 0.000114655140153",
            "extra": "mean: 27.1203838108 msec\nrounds: 37"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-False]",
            "value": 36.9065091691,
            "unit": "iter/sec",
            "range": "stddev: 0.000158607166661",
            "extra": "mean: 27.0954913513 msec\nrounds: 37"
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
          "id": "1fa4171714cce70e209114d3d145ac8d2da1104d",
          "message": "[Deprecation] Warn that from_json without pandas will read missing numbers as NaN (#3086)",
          "timestamp": "2026-10-20T11:02:42-04:00",
          "tree_id": "ef9b143bc6bb41b404fcd104c7800a9e3fd4dfb8",
          "url": "https://github.com/pytorch/tensordict/commit/1fa4171714cce70e209114d3d145ac8d2da1104d"
        },
        "date": 1792510078727,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-2]",
            "value": 4349.14546099,
            "unit": "iter/sec",
            "range": "stddev: 0.00000857238911203",
            "extra": "mean: 229.930226287 usec\nrounds: 137"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-8]",
            "value": 3470.21150915,
            "unit": "iter/sec",
            "range": "stddev: 0.0000120606007574",
            "extra": "mean: 288.166873219 usec\nrounds: 2674"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-32]",
            "value": 1973.95705449,
            "unit": "iter/sec",
            "range": "stddev: 0.0000136590997607",
            "extra": "mean: 506.596634268 usec\nrounds: 1698"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-2]",
            "value": 4392.15245601,
            "unit": "iter/sec",
            "range": "stddev: 0.00000975474001658",
            "extra": "mean: 227.678799863 usec\nrounds: 3058"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-8]",
            "value": 3450.31891175,
            "unit": "iter/sec",
            "range": "stddev: 0.00010254530125",
            "extra": "mean: 289.828281262 usec\nrounds: 3143"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-32]",
            "value": 1771.63411624,
            "unit": "iter/sec",
            "range": "stddev: 0.0000181960798462",
            "extra": "mean: 564.450633926 usec\nrounds: 1609"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-2]",
            "value": 2213.71525053,
            "unit": "iter/sec",
            "range": "stddev: 0.0000157614507693",
            "extra": "mean: 451.729281695 usec\nrounds: 71"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-8]",
            "value": 604.93432896,
            "unit": "iter/sec",
            "range": "stddev: 0.000120940564526",
            "extra": "mean: 1.65307199828 msec\nrounds: 583"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-32]",
            "value": 154.644355398,
            "unit": "iter/sec",
            "range": "stddev: 0.0000733522609275",
            "extra": "mean: 6.46645005197 msec\nrounds: 154"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-2]",
            "value": 2279.98137389,
            "unit": "iter/sec",
            "range": "stddev: 0.0000144915133975",
            "extra": "mean: 438.600074304 usec\nrounds: 2059"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-8]",
            "value": 611.030740057,
            "unit": "iter/sec",
            "range": "stddev: 0.0000308441415957",
            "extra": "mean: 1.63657887311 msec\nrounds: 591"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-32]",
            "value": 155.197072102,
            "unit": "iter/sec",
            "range": "stddev: 0.0000907521083453",
            "extra": "mean: 6.44342052627 msec\nrounds: 152"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-2]",
            "value": 3069.37479885,
            "unit": "iter/sec",
            "range": "stddev: 0.0000129415098195",
            "extra": "mean: 325.799247578 usec\nrounds: 1959"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-8]",
            "value": 831.639521424,
            "unit": "iter/sec",
            "range": "stddev: 0.0000236283532457",
            "extra": "mean: 1.20244405688 msec\nrounds: 791"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-32]",
            "value": 213.388500808,
            "unit": "iter/sec",
            "range": "stddev: 0.0000472933370542",
            "extra": "mean: 4.68628813743 msec\nrounds: 211"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-2]",
            "value": 3065.24787236,
            "unit": "iter/sec",
            "range": "stddev: 0.00000845209799432",
            "extra": "mean: 326.237890585 usec\nrounds: 2879"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-8]",
            "value": 829.48900171,
            "unit": "iter/sec",
            "range": "stddev: 0.0000184839970834",
            "extra": "mean: 1.20556149381 msec\nrounds: 810"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-32]",
            "value": 211.163966873,
            "unit": "iter/sec",
            "range": "stddev: 0.0000501629245444",
            "extra": "mean: 4.73565644181 msec\nrounds: 206"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-2]",
            "value": 3576.28055991,
            "unit": "iter/sec",
            "range": "stddev: 0.0000116232811011",
            "extra": "mean: 279.620120191 usec\nrounds: 2671"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-8]",
            "value": 979.102083862,
            "unit": "iter/sec",
            "range": "stddev: 0.0000188226626826",
            "extra": "mean: 1.02134396043 msec\nrounds: 910"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-32]",
            "value": 250.029095204,
            "unit": "iter/sec",
            "range": "stddev: 0.0000486588253073",
            "extra": "mean: 3.9995345309 msec\nrounds: 243"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-2]",
            "value": 3594.06487657,
            "unit": "iter/sec",
            "range": "stddev: 0.0000101158891222",
            "extra": "mean: 278.236491088 usec\nrounds: 3032"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-8]",
            "value": 968.580601512,
            "unit": "iter/sec",
            "range": "stddev: 0.0000196974561379",
            "extra": "mean: 1.03243859978 msec\nrounds: 952"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-32]",
            "value": 246.463403313,
            "unit": "iter/sec",
            "range": "stddev: 0.0000586456489523",
            "extra": "mean: 4.05739751443 msec\nrounds: 243"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-128]",
            "value": 179.202019055,
            "unit": "iter/sec",
            "range": "stddev: 0.000209321015739",
            "extra": "mean: 5.58029426943 msec\nrounds: 167"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-256]",
            "value": 90.3295928017,
            "unit": "iter/sec",
            "range": "stddev: 0.000607324960433",
            "extra": "mean: 11.0705691123 msec\nrounds: 89"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-128]",
            "value": 177.954561242,
            "unit": "iter/sec",
            "range": "stddev: 0.000113159567246",
            "extra": "mean: 5.61941201743 msec\nrounds: 172"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-256]",
            "value": 90.2107866609,
            "unit": "iter/sec",
            "range": "stddev: 0.000776164784435",
            "extra": "mean: 11.085148872 msec\nrounds: 86"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-128]",
            "value": 12.8149293027,
            "unit": "iter/sec",
            "range": "stddev: 0.00121845450767",
            "extra": "mean: 78.0339849233 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-256]",
            "value": 6.38987290295,
            "unit": "iter/sec",
            "range": "stddev: 0.000748661761376",
            "extra": "mean: 156.497635429 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-128]",
            "value": 12.8096131157,
            "unit": "iter/sec",
            "range": "stddev: 0.00104339952235",
            "extra": "mean: 78.0663702308 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-256]",
            "value": 6.44591352154,
            "unit": "iter/sec",
            "range": "stddev: 0.000919963199302",
            "extra": "mean: 155.137048714 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-128]",
            "value": 23.5306238918,
            "unit": "iter/sec",
            "range": "stddev: 0.000224794521022",
            "extra": "mean: 42.4978107083 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-256]",
            "value": 11.822535362,
            "unit": "iter/sec",
            "range": "stddev: 0.00019067383144",
            "extra": "mean: 84.5842257499 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-128]",
            "value": 23.4516967503,
            "unit": "iter/sec",
            "range": "stddev: 0.000186436315268",
            "extra": "mean: 42.6408379167 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-256]",
            "value": 11.7298276538,
            "unit": "iter/sec",
            "range": "stddev: 0.000268263787209",
            "extra": "mean: 85.2527445 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-128]",
            "value": 27.3701882922,
            "unit": "iter/sec",
            "range": "stddev: 0.000908977356133",
            "extra": "mean: 36.5361023214 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-256]",
            "value": 12.9918913838,
            "unit": "iter/sec",
            "range": "stddev: 0.000915713112763",
            "extra": "mean: 76.9710868463 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-128]",
            "value": 27.4037080286,
            "unit": "iter/sec",
            "range": "stddev: 0.000293306065507",
            "extra": "mean: 36.491412 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-256]",
            "value": 13.7883395566,
            "unit": "iter/sec",
            "range": "stddev: 0.000520725405859",
            "extra": "mean: 72.5250488571 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_mask_many_leaves[cuda]",
            "value": 6228.09878021,
            "unit": "iter/sec",
            "range": "stddev: 0.000016386198268",
            "extra": "mean: 160.562642837 usec\nrounds: 42"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-None]",
            "value": 6556.10561532,
            "unit": "iter/sec",
            "range": "stddev: 0.00000727516379651",
            "extra": "mean: 152.529574518 usec\nrounds: 1342"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-cpu]",
            "value": 6702.54335767,
            "unit": "iter/sec",
            "range": "stddev: 0.00000722609482004",
            "extra": "mean: 149.197095287 usec\nrounds: 4859"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to",
            "value": 23014.5300721,
            "unit": "iter/sec",
            "range": "stddev: 0.00000432822031014",
            "extra": "mean: 43.4508111556 usec\nrounds: 11708"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_nonblocking",
            "value": 23591.6163567,
            "unit": "iter/sec",
            "range": "stddev: 0.00000313899427721",
            "extra": "mean: 42.3879392103 usec\nrounds: 12519"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[add]",
            "value": 10300.0285163,
            "unit": "iter/sec",
            "range": "stddev: 0.00000877707031071",
            "extra": "mean: 97.0871098479 usec\nrounds: 173"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[sub]",
            "value": 10621.6034373,
            "unit": "iter/sec",
            "range": "stddev: 0.00000619931030206",
            "extra": "mean: 94.147743879 usec\nrounds: 7758"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_blocking",
            "value": 54.5396518507,
            "unit": "iter/sec",
            "range": "stddev: 0.0000958578559813",
            "extra": "mean: 18.335283891 msec\nrounds: 55"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_global_sync",
            "value": 93.5942387132,
            "unit": "iter/sec",
            "range": "stddev: 0.0136773213632",
            "extra": "mean: 10.6844183333 msec\nrounds: 108"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_event_sync",
            "value": 97.7780438304,
            "unit": "iter/sec",
            "range": "stddev: 0.000099459974287",
            "extra": "mean: 10.227244899 msec\nrounds: 99"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_default",
            "value": 96.9174327269,
            "unit": "iter/sec",
            "range": "stddev: 0.000146652699981",
            "extra": "mean: 10.3180611771 msec\nrounds: 96"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_cpu_to_cpu_default",
            "value": 3883.6541976,
            "unit": "iter/sec",
            "range": "stddev: 0.0000464836823776",
            "extra": "mean: 257.489454292 usec\nrounds: 3762"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[False-None]",
            "value": 289.111320497,
            "unit": "iter/sec",
            "range": "stddev: 0.0000613893711211",
            "extra": "mean: 3.45887528126 msec\nrounds: 288"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[default-None]",
            "value": 436.619028417,
            "unit": "iter/sec",
            "range": "stddev: 0.000215008030704",
            "extra": "mean: 2.29032619954 msec\nrounds: 441"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[reduce-overhead-None]",
            "value": 444.444516621,
            "unit": "iter/sec",
            "range": "stddev: 0.000310206841626",
            "extra": "mean: 2.24999963461 msec\nrounds: 416"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[False-False-None]",
            "value": 496.83573567,
            "unit": "iter/sec",
            "range": "stddev: 0.0000517038918257",
            "extra": "mean: 2.01273766802 msec\nrounds: 473"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-False-None]",
            "value": 642.729113068,
            "unit": "iter/sec",
            "range": "stddev: 0.0000343270424207",
            "extra": "mean: 1.55586541774 msec\nrounds: 608"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[within-False-None]",
            "value": 194.836494963,
            "unit": "iter/sec",
            "range": "stddev: 0.0000606334019154",
            "extra": "mean: 5.1325086719 msec\nrounds: 192"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-default-None]",
            "value": 114.278579067,
            "unit": "iter/sec",
            "range": "stddev: 0.0000776656360343",
            "extra": "mean: 8.75054632429 msec\nrounds: 111"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[False-False-None]",
            "value": 137.400817034,
            "unit": "iter/sec",
            "range": "stddev: 0.000723247727787",
            "extra": "mean: 7.27797710078 msec\nrounds: 139"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[True-False-None]",
            "value": 262.95079639,
            "unit": "iter/sec",
            "range": "stddev: 0.0000627222985274",
            "extra": "mean: 3.80299285543 msec\nrounds: 256"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[within-False-None]",
            "value": 111.79012641,
            "unit": "iter/sec",
            "range": "stddev: 0.000116089360539",
            "extra": "mean: 8.94533383328 msec\nrounds: 108"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 13.5365504541,
            "unit": "iter/sec",
            "range": "stddev: 0.00229367743512",
            "extra": "mean: 73.8740643998 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_archive",
            "value": 4.42827502135,
            "unit": "iter/sec",
            "range": "stddev: 0.0015690581406",
            "extra": "mean: 225.821566 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_zarr",
            "value": 0.651974804332,
            "unit": "iter/sec",
            "range": "stddev: 0.0453740800224",
            "extra": "mean: 1.533801603 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 42.0395172474,
            "unit": "iter/sec",
            "range": "stddev: 0.000824508961866",
            "extra": "mean: 23.7871427998 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 13.0436476241,
            "unit": "iter/sec",
            "range": "stddev: 0.00448403392596",
            "extra": "mean: 76.6656712 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 8.03895417372,
            "unit": "iter/sec",
            "range": "stddev: 0.125206941233",
            "extra": "mean: 124.3942904 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 40.8492770578,
            "unit": "iter/sec",
            "range": "stddev: 0.00490554496446",
            "extra": "mean: 24.4802374002 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-compile]",
            "value": 4603.02358017,
            "unit": "iter/sec",
            "range": "stddev: 0.000011233586382",
            "extra": "mean: 217.248506896 usec\nrounds: 1740"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-eager]",
            "value": 7726.24848988,
            "unit": "iter/sec",
            "range": "stddev: 0.00000853716898191",
            "extra": "mean: 129.428920298 usec\nrounds: 6537"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-compile]",
            "value": 6261.23406286,
            "unit": "iter/sec",
            "range": "stddev: 0.00000987365549413",
            "extra": "mean: 159.712923996 usec\nrounds: 2829"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-eager]",
            "value": 4701.06289498,
            "unit": "iter/sec",
            "range": "stddev: 0.0000654512446722",
            "extra": "mean: 212.717851758 usec\nrounds: 5181"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-compile]",
            "value": 18011.8067197,
            "unit": "iter/sec",
            "range": "stddev: 0.0000157636333419",
            "extra": "mean: 55.5191389495 usec\nrounds: 15286"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-eager]",
            "value": 30682.1546622,
            "unit": "iter/sec",
            "range": "stddev: 0.00000256287882344",
            "extra": "mean: 32.5922351611 usec\nrounds: 25693"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-compile]",
            "value": 59915.1913574,
            "unit": "iter/sec",
            "range": "stddev: 0.00000471889813957",
            "extra": "mean: 16.6902579687 usec\nrounds: 34915"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-eager]",
            "value": 15678.2104056,
            "unit": "iter/sec",
            "range": "stddev: 0.0000127848824841",
            "extra": "mean: 63.7827898804 usec\nrounds: 14777"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-compile]",
            "value": 3634.1910287,
            "unit": "iter/sec",
            "range": "stddev: 0.0000155706642577",
            "extra": "mean: 275.164401679 usec\nrounds: 1307"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-eager]",
            "value": 5232.80364822,
            "unit": "iter/sec",
            "range": "stddev: 0.00000982877856618",
            "extra": "mean: 191.102144706 usec\nrounds: 4948"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-compile]",
            "value": 5741.96373591,
            "unit": "iter/sec",
            "range": "stddev: 0.00000937217986093",
            "extra": "mean: 174.156446469 usec\nrounds: 3288"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-eager]",
            "value": 15767.0259458,
            "unit": "iter/sec",
            "range": "stddev: 0.00000496590836917",
            "extra": "mean: 63.423501898 usec\nrounds: 13423"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-compile]",
            "value": 4026.45079977,
            "unit": "iter/sec",
            "range": "stddev: 0.00000960582041642",
            "extra": "mean: 248.357685151 usec\nrounds: 1353"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-eager]",
            "value": 1909.65263959,
            "unit": "iter/sec",
            "range": "stddev: 0.0000219992168312",
            "extra": "mean: 523.655443545 usec\nrounds: 1860"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-eager]",
            "value": 4275.64745132,
            "unit": "iter/sec",
            "range": "stddev: 0.0000106260811579",
            "extra": "mean: 233.882706978 usec\nrounds: 4027"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-compile]",
            "value": 3610.19669895,
            "unit": "iter/sec",
            "range": "stddev: 0.0000128181846734",
            "extra": "mean: 276.993217653 usec\nrounds: 1314"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-eager]",
            "value": 12789.2086504,
            "unit": "iter/sec",
            "range": "stddev: 0.00000665125175407",
            "extra": "mean: 78.1909207469 usec\nrounds: 11129"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-compile]",
            "value": 5705.9612803,
            "unit": "iter/sec",
            "range": "stddev: 0.00000907227703098",
            "extra": "mean: 175.255307717 usec\nrounds: 3162"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-eager]",
            "value": 2153.54932096,
            "unit": "iter/sec",
            "range": "stddev: 0.0000189504053931",
            "extra": "mean: 464.349708766 usec\nrounds: 2146"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-compile]",
            "value": 4032.17324651,
            "unit": "iter/sec",
            "range": "stddev: 0.0000102472260072",
            "extra": "mean: 248.005216756 usec\nrounds: 1384"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-compile]",
            "value": 33439.2764094,
            "unit": "iter/sec",
            "range": "stddev: 0.00000283272750497",
            "extra": "mean: 29.9049533177 usec\nrounds: 20929"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-eager]",
            "value": 53552.3824763,
            "unit": "iter/sec",
            "range": "stddev: 0.00000180776055573",
            "extra": "mean: 18.673305533 usec\nrounds: 46532"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-compile]",
            "value": 45649.554629,
            "unit": "iter/sec",
            "range": "stddev: 0.00000216654343161",
            "extra": "mean: 21.906018758 usec\nrounds: 24893"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-eager]",
            "value": 19064.9456064,
            "unit": "iter/sec",
            "range": "stddev: 0.000011500455093",
            "extra": "mean: 52.4522870741 usec\nrounds: 16661"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-compile]",
            "value": 4018.57917012,
            "unit": "iter/sec",
            "range": "stddev: 0.000225587095955",
            "extra": "mean: 248.844170455 usec\nrounds: 88"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-eager]",
            "value": 314.64352884,
            "unit": "iter/sec",
            "range": "stddev: 0.0000678847124744",
            "extra": "mean: 3.17819979863 msec\nrounds: 298"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-compile]",
            "value": 4910.90940703,
            "unit": "iter/sec",
            "range": "stddev: 0.000210203498902",
            "extra": "mean: 203.628272712 usec\nrounds: 88"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-eager]",
            "value": 358.905034673,
            "unit": "iter/sec",
            "range": "stddev: 0.0000429548299571",
            "extra": "mean: 2.78625236035 msec\nrounds: 358"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-compile]",
            "value": 6101.31216694,
            "unit": "iter/sec",
            "range": "stddev: 0.0000072314708414",
            "extra": "mean: 163.899169988 usec\nrounds: 3806"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-eager]",
            "value": 18367.2083289,
            "unit": "iter/sec",
            "range": "stddev: 0.0000124290800054",
            "extra": "mean: 54.4448553145 usec\nrounds: 14590"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-compile]",
            "value": 7481.43001125,
            "unit": "iter/sec",
            "range": "stddev: 0.00000703505220759",
            "extra": "mean: 133.664285905 usec\nrounds: 4351"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-eager]",
            "value": 21437.1213667,
            "unit": "iter/sec",
            "range": "stddev: 0.0000118470111823",
            "extra": "mean: 46.6480542277 usec\nrounds: 17500"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-compile]",
            "value": 7423.5043123,
            "unit": "iter/sec",
            "range": "stddev: 0.00000687496101971",
            "extra": "mean: 134.707270035 usec\nrounds: 6773"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-eager]",
            "value": 21917.7250857,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113024784258",
            "extra": "mean: 45.6251730547 usec\nrounds: 17873"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-compile]",
            "value": 11562.3848295,
            "unit": "iter/sec",
            "range": "stddev: 0.00000585015100439",
            "extra": "mean: 86.4873479604 usec\nrounds: 9185"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-eager]",
            "value": 46721.7020254,
            "unit": "iter/sec",
            "range": "stddev: 0.0000107975286235",
            "extra": "mean: 21.4033298585 usec\nrounds: 32635"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-compile]",
            "value": 15716.5399819,
            "unit": "iter/sec",
            "range": "stddev: 0.00000424030985565",
            "extra": "mean: 63.6272360934 usec\nrounds: 12474"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-eager]",
            "value": 47436.0721832,
            "unit": "iter/sec",
            "range": "stddev: 0.0000109215507436",
            "extra": "mean: 21.0810034216 usec\nrounds: 34782"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-compile]",
            "value": 15470.0823454,
            "unit": "iter/sec",
            "range": "stddev: 0.00000443950635525",
            "extra": "mean: 64.6408970342 usec\nrounds: 13024"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-eager]",
            "value": 47304.2395445,
            "unit": "iter/sec",
            "range": "stddev: 0.0000109765810377",
            "extra": "mean: 21.1397542721 usec\nrounds: 36442"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-compile]",
            "value": 11493.992506,
            "unit": "iter/sec",
            "range": "stddev: 0.00000522766278266",
            "extra": "mean: 87.0019707668 usec\nrounds: 10226"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-eager]",
            "value": 47417.0890505,
            "unit": "iter/sec",
            "range": "stddev: 0.0000106548718021",
            "extra": "mean: 21.0894430684 usec\nrounds: 35726"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-compile]",
            "value": 15527.6116434,
            "unit": "iter/sec",
            "range": "stddev: 0.00000401776475586",
            "extra": "mean: 64.4014046055 usec\nrounds: 13428"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-eager]",
            "value": 47388.2225709,
            "unit": "iter/sec",
            "range": "stddev: 0.0000109096781652",
            "extra": "mean: 21.1022896776 usec\nrounds: 38864"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-compile]",
            "value": 15521.3885029,
            "unit": "iter/sec",
            "range": "stddev: 0.00000416210412984",
            "extra": "mean: 64.4272256836 usec\nrounds: 13337"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-eager]",
            "value": 47449.0750773,
            "unit": "iter/sec",
            "range": "stddev: 0.0000108269640408",
            "extra": "mean: 21.0752264058 usec\nrounds: 38051"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-eager]",
            "value": 32544.7117227,
            "unit": "iter/sec",
            "range": "stddev: 0.00000260760294064",
            "extra": "mean: 30.7269583002 usec\nrounds: 23261"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-compile]",
            "value": 6833.29159308,
            "unit": "iter/sec",
            "range": "stddev: 0.00000705202185352",
            "extra": "mean: 146.342357322 usec\nrounds: 4142"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-eager]",
            "value": 2667.47905633,
            "unit": "iter/sec",
            "range": "stddev: 0.0000146225742187",
            "extra": "mean: 374.885792497 usec\nrounds: 2395"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-compile]",
            "value": 6433.22995416,
            "unit": "iter/sec",
            "range": "stddev: 0.00000835480407475",
            "extra": "mean: 155.442912367 usec\nrounds: 3526"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[eager]",
            "value": 6038.23501981,
            "unit": "iter/sec",
            "range": "stddev: 0.00000689292191654",
            "extra": "mean: 165.611308059 usec\nrounds: 5882"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[compile]",
            "value": 5596.51825652,
            "unit": "iter/sec",
            "range": "stddev: 0.00000843595882579",
            "extra": "mean: 178.682522627 usec\nrounds: 3140"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[eager]",
            "value": 5601.75340935,
            "unit": "iter/sec",
            "range": "stddev: 0.00000777432307228",
            "extra": "mean: 178.515533785 usec\nrounds: 5062"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[compile]",
            "value": 4970.86092358,
            "unit": "iter/sec",
            "range": "stddev: 0.00000760748071951",
            "extra": "mean: 201.172395562 usec\nrounds: 3289"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-eager]",
            "value": 109723.890049,
            "unit": "iter/sec",
            "range": "stddev: 8.55411145609e-7",
            "extra": "mean: 9.11378551699 usec\nrounds: 84389"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-compile]",
            "value": 40907.6651287,
            "unit": "iter/sec",
            "range": "stddev: 0.00000825140139431",
            "extra": "mean: 24.4452964219 usec\nrounds: 23419"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-eager]",
            "value": 66005.9113974,
            "unit": "iter/sec",
            "range": "stddev: 0.00000105621797437",
            "extra": "mean: 15.1501582029 usec\nrounds: 47369"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-compile]",
            "value": 34032.3421301,
            "unit": "iter/sec",
            "range": "stddev: 0.00000304617554839",
            "extra": "mean: 29.3838136728 usec\nrounds: 20829"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-eager]",
            "value": 37152.2483587,
            "unit": "iter/sec",
            "range": "stddev: 0.00000152745989616",
            "extra": "mean: 26.9162714015 usec\nrounds: 32030"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-compile]",
            "value": 22864.3919736,
            "unit": "iter/sec",
            "range": "stddev: 0.00000354300969544",
            "extra": "mean: 43.7361291371 usec\nrounds: 16711"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[eager]",
            "value": 36079.4690343,
            "unit": "iter/sec",
            "range": "stddev: 0.00000174426389688",
            "extra": "mean: 27.7165941397 usec\nrounds: 27374"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[compile]",
            "value": 8299.72806539,
            "unit": "iter/sec",
            "range": "stddev: 0.00000546459367748",
            "extra": "mean: 120.485875214 usec\nrounds: 6964"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[eager]",
            "value": 30162.8536659,
            "unit": "iter/sec",
            "range": "stddev: 0.00000298408401675",
            "extra": "mean: 33.1533617832 usec\nrounds: 20280"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile]",
            "value": 8929.7407495,
            "unit": "iter/sec",
            "range": "stddev: 0.0000308273902981",
            "extra": "mean: 111.985333959 usec\nrounds: 5767"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile-overhead]",
            "value": 5145.17134012,
            "unit": "iter/sec",
            "range": "stddev: 0.0000153750768012",
            "extra": "mean: 194.356987143 usec\nrounds: 310"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[eager]",
            "value": 3619.36933447,
            "unit": "iter/sec",
            "range": "stddev: 0.0000104283691316",
            "extra": "mean: 276.291228551 usec\nrounds: 3312"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile]",
            "value": 3065.02143848,
            "unit": "iter/sec",
            "range": "stddev: 0.0000105579981162",
            "extra": "mean: 326.261991987 usec\nrounds: 2248"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile-overhead]",
            "value": 269.261777793,
            "unit": "iter/sec",
            "range": "stddev: 0.00160614981349",
            "extra": "mean: 3.71385797196 msec\nrounds: 927"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[False]",
            "value": 6217.67699224,
            "unit": "iter/sec",
            "range": "stddev: 0.00000717669997667",
            "extra": "mean: 160.831770651 usec\nrounds: 4988"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[True]",
            "value": 6051.73072342,
            "unit": "iter/sec",
            "range": "stddev: 0.0000122673073757",
            "extra": "mean: 165.241985426 usec\nrounds: 5214"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[eager]",
            "value": 634.423090166,
            "unit": "iter/sec",
            "range": "stddev: 0.000024856586729",
            "extra": "mean: 1.57623518989 msec\nrounds: 616"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile]",
            "value": 669.685886493,
            "unit": "iter/sec",
            "range": "stddev: 0.0000667675510048",
            "extra": "mean: 1.49323738214 msec\nrounds: 526"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile-overhead]",
            "value": 948.974244802,
            "unit": "iter/sec",
            "range": "stddev: 0.000144368628952",
            "extra": "mean: 1.05376937833 msec\nrounds: 74"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[eager]",
            "value": 10015.773279,
            "unit": "iter/sec",
            "range": "stddev: 0.00000804280596788",
            "extra": "mean: 99.8425156145 usec\nrounds: 8291"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile]",
            "value": 7774.53784127,
            "unit": "iter/sec",
            "range": "stddev: 0.00000929375494764",
            "extra": "mean: 128.625009025 usec\nrounds: 6534"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile-overhead]",
            "value": 4826.93364398,
            "unit": "iter/sec",
            "range": "stddev: 0.00000786593035615",
            "extra": "mean: 207.170861204 usec\nrounds: 3242"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[eager]",
            "value": 2451.05533528,
            "unit": "iter/sec",
            "range": "stddev: 0.000012780636292",
            "extra": "mean: 407.987525049 usec\nrounds: 2316"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile]",
            "value": 2898.05804107,
            "unit": "iter/sec",
            "range": "stddev: 0.0000106539623495",
            "extra": "mean: 345.058651631 usec\nrounds: 2417"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile-overhead]",
            "value": 3251.83661278,
            "unit": "iter/sec",
            "range": "stddev: 0.00000980856528818",
            "extra": "mean: 307.518525399 usec\nrounds: 2166"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-eager]",
            "value": 1186.94242837,
            "unit": "iter/sec",
            "range": "stddev: 0.000016241223373",
            "extra": "mean: 842.50084595 usec\nrounds: 1136"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile]",
            "value": 1406.85596711,
            "unit": "iter/sec",
            "range": "stddev: 0.0000150311973908",
            "extra": "mean: 710.804818246 usec\nrounds: 1348"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile-overhead]",
            "value": 2106.71807985,
            "unit": "iter/sec",
            "range": "stddev: 0.0000198420816462",
            "extra": "mean: 474.671959939 usec\nrounds: 1323"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-eager]",
            "value": 757.247669844,
            "unit": "iter/sec",
            "range": "stddev: 0.000027844024056",
            "extra": "mean: 1.32057190774 msec\nrounds: 737"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile]",
            "value": 1352.80593193,
            "unit": "iter/sec",
            "range": "stddev: 0.0000364634707002",
            "extra": "mean: 739.204328128 usec\nrounds: 1283"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile-overhead]",
            "value": 2006.11466027,
            "unit": "iter/sec",
            "range": "stddev: 0.0000130361326603",
            "extra": "mean: 498.475994322 usec\nrounds: 1238"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-eager]",
            "value": 1176.73290964,
            "unit": "iter/sec",
            "range": "stddev: 0.0000170820905243",
            "extra": "mean: 849.810515032 usec\nrounds: 1097"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile]",
            "value": 1409.70509811,
            "unit": "iter/sec",
            "range": "stddev: 0.0000188747554558",
            "extra": "mean: 709.368222717 usec\nrounds: 1338"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile-overhead]",
            "value": 2186.57446563,
            "unit": "iter/sec",
            "range": "stddev: 0.0000155543283335",
            "extra": "mean: 457.336356808 usec\nrounds: 1079"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-eager]",
            "value": 729.881408992,
            "unit": "iter/sec",
            "range": "stddev: 0.0000272753176448",
            "extra": "mean: 1.37008558881 msec\nrounds: 715"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile]",
            "value": 1214.33724792,
            "unit": "iter/sec",
            "range": "stddev: 0.0000146906968094",
            "extra": "mean: 823.494463099 usec\nrounds: 1179"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile-overhead]",
            "value": 1306.51618982,
            "unit": "iter/sec",
            "range": "stddev: 0.000100757066076",
            "extra": "mean: 765.394265905 usec\nrounds: 1038"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[eager]",
            "value": 340.617946876,
            "unit": "iter/sec",
            "range": "stddev: 0.0000450694436693",
            "extra": "mean: 2.93584060726 msec\nrounds: 331"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile]",
            "value": 1310.1006939,
            "unit": "iter/sec",
            "range": "stddev: 0.0000182959721484",
            "extra": "mean: 763.300107126 usec\nrounds: 1036"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile-overhead]",
            "value": 1914.06000254,
            "unit": "iter/sec",
            "range": "stddev: 0.00003130092574",
            "extra": "mean: 522.449661283 usec\nrounds: 124"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1027.86095242,
            "unit": "iter/sec",
            "range": "stddev: 0.0000367364954535",
            "extra": "mean: 972.894239877 usec\nrounds: 963"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1031.07431898,
            "unit": "iter/sec",
            "range": "stddev: 0.0000373628825036",
            "extra": "mean: 969.862192854 usec\nrounds: 1006"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1120.99317572,
            "unit": "iter/sec",
            "range": "stddev: 0.0000365785790077",
            "extra": "mean: 892.066090732 usec\nrounds: 1102"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1122.03645458,
            "unit": "iter/sec",
            "range": "stddev: 0.0000386659972372",
            "extra": "mean: 891.236640233 usec\nrounds: 1123"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-True]",
            "value": 36.8062429929,
            "unit": "iter/sec",
            "range": "stddev: 0.000461749175993",
            "extra": "mean: 27.1693038649 msec\nrounds: 37"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-False]",
            "value": 37.253388421,
            "unit": "iter/sec",
            "range": "stddev: 0.000214814646042",
            "extra": "mean: 26.8431958108 msec\nrounds: 37"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-True]",
            "value": 37.1412506442,
            "unit": "iter/sec",
            "range": "stddev: 0.000320703862024",
            "extra": "mean: 26.9242414473 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-False]",
            "value": 37.5398851894,
            "unit": "iter/sec",
            "range": "stddev: 0.0000652658780216",
            "extra": "mean: 26.6383340001 msec\nrounds: 38"
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
          "id": "dce62f0499b170e1141cc5b869792129d9f67566",
          "message": "[Typing] Ship a py.typed marker, so that mypy checks code that uses an installed tensordict (#3097)",
          "timestamp": "2026-10-20T11:12:21-04:00",
          "tree_id": "7a825058318e3c77d9b62a718dbde5dfe78d96a9",
          "url": "https://github.com/pytorch/tensordict/commit/dce62f0499b170e1141cc5b869792129d9f67566"
        },
        "date": 1792510486852,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-2]",
            "value": 4561.3285827,
            "unit": "iter/sec",
            "range": "stddev: 0.00000638845494073",
            "extra": "mean: 219.234370397 usec\nrounds: 135"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-8]",
            "value": 3600.23800393,
            "unit": "iter/sec",
            "range": "stddev: 0.00000677967469793",
            "extra": "mean: 277.759414491 usec\nrounds: 3093"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-32]",
            "value": 2033.10732554,
            "unit": "iter/sec",
            "range": "stddev: 0.00000962742559997",
            "extra": "mean: 491.857949377 usec\nrounds: 1837"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-2]",
            "value": 4586.47314896,
            "unit": "iter/sec",
            "range": "stddev: 0.00000549206371041",
            "extra": "mean: 218.0324549 usec\nrounds: 3992"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-8]",
            "value": 3601.60367864,
            "unit": "iter/sec",
            "range": "stddev: 0.00000678903299243",
            "extra": "mean: 277.65409224 usec\nrounds: 3198"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-32]",
            "value": 1799.20188385,
            "unit": "iter/sec",
            "range": "stddev: 0.00000684646675462",
            "extra": "mean: 555.801996973 usec\nrounds: 1651"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-2]",
            "value": 2200.77613437,
            "unit": "iter/sec",
            "range": "stddev: 0.0000136232281419",
            "extra": "mean: 454.385152756 usec\nrounds: 72"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-8]",
            "value": 605.160850528,
            "unit": "iter/sec",
            "range": "stddev: 0.0000294475955296",
            "extra": "mean: 1.65245322649 msec\nrounds: 574"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-32]",
            "value": 154.649294063,
            "unit": "iter/sec",
            "range": "stddev: 0.000100749891193",
            "extra": "mean: 6.46624354841 msec\nrounds: 155"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-2]",
            "value": 2209.12634986,
            "unit": "iter/sec",
            "range": "stddev: 0.0000405947424254",
            "extra": "mean: 452.667634906 usec\nrounds: 2120"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-8]",
            "value": 571.050457657,
            "unit": "iter/sec",
            "range": "stddev: 0.0000538758648804",
            "extra": "mean: 1.75115874016 msec\nrounds: 535"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-32]",
            "value": 144.950309728,
            "unit": "iter/sec",
            "range": "stddev: 0.0000627433282424",
            "extra": "mean: 6.89891592418 msec\nrounds: 145"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-2]",
            "value": 3102.73847384,
            "unit": "iter/sec",
            "range": "stddev: 0.00000990309222282",
            "extra": "mean: 322.29593581 usec\nrounds: 2399"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-8]",
            "value": 842.596919673,
            "unit": "iter/sec",
            "range": "stddev: 0.0000187647614612",
            "extra": "mean: 1.18680709204 msec\nrounds: 804"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-32]",
            "value": 215.236310975,
            "unit": "iter/sec",
            "range": "stddev: 0.0000269367449389",
            "extra": "mean: 4.64605621361 msec\nrounds: 206"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-2]",
            "value": 3116.68717723,
            "unit": "iter/sec",
            "range": "stddev: 0.00000627365639827",
            "extra": "mean: 320.853503459 usec\nrounds: 2898"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-8]",
            "value": 835.425557567,
            "unit": "iter/sec",
            "range": "stddev: 0.0000169538055374",
            "extra": "mean: 1.19699474231 msec\nrounds: 815"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-32]",
            "value": 213.857186112,
            "unit": "iter/sec",
            "range": "stddev: 0.0000312290239258",
            "extra": "mean: 4.67601775829 msec\nrounds: 211"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-2]",
            "value": 3621.75704468,
            "unit": "iter/sec",
            "range": "stddev: 0.00000969052557388",
            "extra": "mean: 276.109078457 usec\nrounds: 3059"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-8]",
            "value": 974.390321825,
            "unit": "iter/sec",
            "range": "stddev: 0.0000121730703143",
            "extra": "mean: 1.02628277149 msec\nrounds: 954"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-32]",
            "value": 246.028822665,
            "unit": "iter/sec",
            "range": "stddev: 0.0000882674098214",
            "extra": "mean: 4.06456442448 msec\nrounds: 245"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-2]",
            "value": 3617.02367186,
            "unit": "iter/sec",
            "range": "stddev: 0.00000807469936274",
            "extra": "mean: 276.470405151 usec\nrounds: 3184"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-8]",
            "value": 964.200889307,
            "unit": "iter/sec",
            "range": "stddev: 0.0000147737849208",
            "extra": "mean: 1.03712826973 msec\nrounds: 912"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-32]",
            "value": 244.995809644,
            "unit": "iter/sec",
            "range": "stddev: 0.000049549997766",
            "extra": "mean: 4.08170246443 msec\nrounds: 239"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-128]",
            "value": 187.272459293,
            "unit": "iter/sec",
            "range": "stddev: 0.0000539366119749",
            "extra": "mean: 5.33981346631 msec\nrounds: 178"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-256]",
            "value": 98.9048474567,
            "unit": "iter/sec",
            "range": "stddev: 0.0000859955057663",
            "extra": "mean: 10.1107278937 msec\nrounds: 94"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-128]",
            "value": 180.264519222,
            "unit": "iter/sec",
            "range": "stddev: 0.000103404693726",
            "extra": "mean: 5.54740336209 msec\nrounds: 174"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-256]",
            "value": 94.3904539789,
            "unit": "iter/sec",
            "range": "stddev: 0.0000999397240125",
            "extra": "mean: 10.5942916666 msec\nrounds: 87"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-128]",
            "value": 13.2573345519,
            "unit": "iter/sec",
            "range": "stddev: 0.000540549168079",
            "extra": "mean: 75.4299437858 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-256]",
            "value": 6.39980207569,
            "unit": "iter/sec",
            "range": "stddev: 0.00389425906183",
            "extra": "mean: 156.254832286 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-128]",
            "value": 12.97545584,
            "unit": "iter/sec",
            "range": "stddev: 0.000405035628793",
            "extra": "mean: 77.0685833572 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-256]",
            "value": 6.56371363064,
            "unit": "iter/sec",
            "range": "stddev: 0.000814296768264",
            "extra": "mean: 152.352777143 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-128]",
            "value": 23.6456228611,
            "unit": "iter/sec",
            "range": "stddev: 0.000110748343106",
            "extra": "mean: 42.2911253333 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-256]",
            "value": 11.8456542173,
            "unit": "iter/sec",
            "range": "stddev: 0.000110521372504",
            "extra": "mean: 84.4191449166 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-128]",
            "value": 23.5250285214,
            "unit": "iter/sec",
            "range": "stddev: 0.000235553181838",
            "extra": "mean: 42.5079187083 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-256]",
            "value": 11.7273573076,
            "unit": "iter/sec",
            "range": "stddev: 0.000500529004123",
            "extra": "mean: 85.2707028334 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-128]",
            "value": 26.988516628,
            "unit": "iter/sec",
            "range": "stddev: 0.00028892588403",
            "extra": "mean: 37.0527959644 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-256]",
            "value": 13.4927636227,
            "unit": "iter/sec",
            "range": "stddev: 0.00184272733299",
            "extra": "mean: 74.1138011428 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-128]",
            "value": 27.4015414485,
            "unit": "iter/sec",
            "range": "stddev: 0.000138965850314",
            "extra": "mean: 36.4942972964 msec\nrounds: 27"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-256]",
            "value": 13.6904958793,
            "unit": "iter/sec",
            "range": "stddev: 0.000309297882468",
            "extra": "mean: 73.0433732141 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_mask_many_leaves[cuda]",
            "value": 6563.61686397,
            "unit": "iter/sec",
            "range": "stddev: 0.0000152640024343",
            "extra": "mean: 152.355023263 usec\nrounds: 43"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-None]",
            "value": 6869.99614009,
            "unit": "iter/sec",
            "range": "stddev: 0.00000422073973732",
            "extra": "mean: 145.560489352 usec\nrounds: 1455"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-cpu]",
            "value": 6955.89725325,
            "unit": "iter/sec",
            "range": "stddev: 0.00000478343620395",
            "extra": "mean: 143.762905574 usec\nrounds: 4829"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to",
            "value": 23407.968234,
            "unit": "iter/sec",
            "range": "stddev: 0.00000205312776998",
            "extra": "mean: 42.7204954314 usec\nrounds: 11812"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_nonblocking",
            "value": 23939.8461906,
            "unit": "iter/sec",
            "range": "stddev: 0.00000201564119108",
            "extra": "mean: 41.7713627748 usec\nrounds: 12738"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[add]",
            "value": 10681.1837316,
            "unit": "iter/sec",
            "range": "stddev: 0.00000738875210488",
            "extra": "mean: 93.622582022 usec\nrounds: 189"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[sub]",
            "value": 10940.3577857,
            "unit": "iter/sec",
            "range": "stddev: 0.00000348693578986",
            "extra": "mean: 91.4046889131 usec\nrounds: 6892"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_blocking",
            "value": 40.99846412,
            "unit": "iter/sec",
            "range": "stddev: 0.000805581954133",
            "extra": "mean: 24.3911576071 msec\nrounds: 56"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_global_sync",
            "value": 95.5365070567,
            "unit": "iter/sec",
            "range": "stddev: 0.0118820609649",
            "extra": "mean: 10.4672028611 msec\nrounds: 108"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_event_sync",
            "value": 96.9505687478,
            "unit": "iter/sec",
            "range": "stddev: 0.000138640227118",
            "extra": "mean: 10.3145346429 msec\nrounds: 98"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_default",
            "value": 94.4176345959,
            "unit": "iter/sec",
            "range": "stddev: 0.000184891487566",
            "extra": "mean: 10.5912418192 msec\nrounds: 94"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_cpu_to_cpu_default",
            "value": 3846.32364233,
            "unit": "iter/sec",
            "range": "stddev: 0.00000466494878577",
            "extra": "mean: 259.988522285 usec\nrounds: 3793"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[False-None]",
            "value": 294.870246771,
            "unit": "iter/sec",
            "range": "stddev: 0.0000575325037827",
            "extra": "mean: 3.39132215254 msec\nrounds: 295"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[default-None]",
            "value": 247.167762663,
            "unit": "iter/sec",
            "range": "stddev: 0.03786674277",
            "extra": "mean: 4.04583506047 msec\nrounds: 430"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[reduce-overhead-None]",
            "value": 451.791525853,
            "unit": "iter/sec",
            "range": "stddev: 0.000253002674613",
            "extra": "mean: 2.21341026287 msec\nrounds: 407"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[False-False-None]",
            "value": 490.692707998,
            "unit": "iter/sec",
            "range": "stddev: 0.0000419455717733",
            "extra": "mean: 2.03793531818 msec\nrounds: 484"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-False-None]",
            "value": 639.787595638,
            "unit": "iter/sec",
            "range": "stddev: 0.0000519294569387",
            "extra": "mean: 1.5630187375 msec\nrounds: 640"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[within-False-None]",
            "value": 196.073308962,
            "unit": "iter/sec",
            "range": "stddev: 0.000051392513279",
            "extra": "mean: 5.10013323739 msec\nrounds: 198"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-default-None]",
            "value": 115.759836454,
            "unit": "iter/sec",
            "range": "stddev: 0.000186272220317",
            "extra": "mean: 8.63857474782 msec\nrounds: 115"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[False-False-None]",
            "value": 138.725784972,
            "unit": "iter/sec",
            "range": "stddev: 0.00015099146296",
            "extra": "mean: 7.20846524823 msec\nrounds: 141"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[True-False-None]",
            "value": 255.85810735,
            "unit": "iter/sec",
            "range": "stddev: 0.000100989344103",
            "extra": "mean: 3.90841631075 msec\nrounds: 251"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[within-False-None]",
            "value": 110.087519579,
            "unit": "iter/sec",
            "range": "stddev: 0.000243154839651",
            "extra": "mean: 9.08368181811 msec\nrounds: 110"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 11.8331138653,
            "unit": "iter/sec",
            "range": "stddev: 0.000375630479471",
            "extra": "mean: 84.5086096004 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_archive",
            "value": 4.6006128445,
            "unit": "iter/sec",
            "range": "stddev: 0.00169832695177",
            "extra": "mean: 217.3623458 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_zarr",
            "value": 0.450422031296,
            "unit": "iter/sec",
            "range": "stddev: 0.117787907852",
            "extra": "mean: 2.2201400698 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 41.9125891332,
            "unit": "iter/sec",
            "range": "stddev: 0.000103786483991",
            "extra": "mean: 23.8591797997 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.49118193177,
            "unit": "iter/sec",
            "range": "stddev: 0.0475539407614",
            "extra": "mean: 105.3609558 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 11.1524544991,
            "unit": "iter/sec",
            "range": "stddev: 0.0013483193556",
            "extra": "mean: 89.66636 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 44.1112936054,
            "unit": "iter/sec",
            "range": "stddev: 0.000593975459395",
            "extra": "mean: 22.6699314 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-compile]",
            "value": 4743.07378981,
            "unit": "iter/sec",
            "range": "stddev: 0.00000851222434006",
            "extra": "mean: 210.833742909 usec\nrounds: 1793"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-eager]",
            "value": 8058.59808759,
            "unit": "iter/sec",
            "range": "stddev: 0.00000620489604488",
            "extra": "mean: 124.091062631 usec\nrounds: 7073"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-compile]",
            "value": 6567.2970752,
            "unit": "iter/sec",
            "range": "stddev: 0.00000836818182264",
            "extra": "mean: 152.269645876 usec\nrounds: 2598"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-eager]",
            "value": 5687.94072991,
            "unit": "iter/sec",
            "range": "stddev: 0.0000160587663509",
            "extra": "mean: 175.810552093 usec\nrounds: 5336"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-compile]",
            "value": 19639.2028262,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032438330184",
            "extra": "mean: 50.9185636937 usec\nrounds: 15111"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-eager]",
            "value": 31642.1036223,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017567422616",
            "extra": "mean: 31.603461386 usec\nrounds: 25961"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-compile]",
            "value": 64608.7034679,
            "unit": "iter/sec",
            "range": "stddev: 0.00000172371667686",
            "extra": "mean: 15.4777908599 usec\nrounds: 38926"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-eager]",
            "value": 15807.9735726,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113439323201",
            "extra": "mean: 63.259215067 usec\nrounds: 14921"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-compile]",
            "value": 3755.71513282,
            "unit": "iter/sec",
            "range": "stddev: 0.00000737189236705",
            "extra": "mean: 266.260875662 usec\nrounds: 1319"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-eager]",
            "value": 5306.68441629,
            "unit": "iter/sec",
            "range": "stddev: 0.00000579202478508",
            "extra": "mean: 188.44158076 usec\nrounds: 5083"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-compile]",
            "value": 5999.79675287,
            "unit": "iter/sec",
            "range": "stddev: 0.00000505322321579",
            "extra": "mean: 166.672312611 usec\nrounds: 3391"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-eager]",
            "value": 16246.7827994,
            "unit": "iter/sec",
            "range": "stddev: 0.00000701425075573",
            "extra": "mean: 61.5506474324 usec\nrounds: 13949"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-compile]",
            "value": 4176.47664539,
            "unit": "iter/sec",
            "range": "stddev: 0.00000736830331755",
            "extra": "mean: 239.436272463 usec\nrounds: 1369"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-eager]",
            "value": 1882.57814383,
            "unit": "iter/sec",
            "range": "stddev: 0.000015973402933",
            "extra": "mean: 531.186449432 usec\nrounds: 1720"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-eager]",
            "value": 4336.49419143,
            "unit": "iter/sec",
            "range": "stddev: 0.00000641642952208",
            "extra": "mean: 230.601023743 usec\nrounds: 1558"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-compile]",
            "value": 3756.8738975,
            "unit": "iter/sec",
            "range": "stddev: 0.00000545098369517",
            "extra": "mean: 266.178750547 usec\nrounds: 1355"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-eager]",
            "value": 13302.7853022,
            "unit": "iter/sec",
            "range": "stddev: 0.00000276180113554",
            "extra": "mean: 75.1722272657 usec\nrounds: 11700"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-compile]",
            "value": 5972.44842848,
            "unit": "iter/sec",
            "range": "stddev: 0.00000427160026357",
            "extra": "mean: 167.435518611 usec\nrounds: 3357"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-eager]",
            "value": 2174.71595819,
            "unit": "iter/sec",
            "range": "stddev: 0.0000178969437496",
            "extra": "mean: 459.830165974 usec\nrounds: 2151"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-compile]",
            "value": 4204.31730616,
            "unit": "iter/sec",
            "range": "stddev: 0.00000580731678405",
            "extra": "mean: 237.850744171 usec\nrounds: 1376"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-compile]",
            "value": 25160.0291607,
            "unit": "iter/sec",
            "range": "stddev: 0.0000108869863868",
            "extra": "mean: 39.7455819154 usec\nrounds: 22139"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-eager]",
            "value": 54193.6777406,
            "unit": "iter/sec",
            "range": "stddev: 8.80094482789e-7",
            "extra": "mean: 18.4523369089 usec\nrounds: 46621"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-compile]",
            "value": 46673.0113393,
            "unit": "iter/sec",
            "range": "stddev: 0.00000192714125141",
            "extra": "mean: 21.4256584545 usec\nrounds: 28418"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-eager]",
            "value": 20102.1937735,
            "unit": "iter/sec",
            "range": "stddev: 0.0000111371690654",
            "extra": "mean: 49.7458143758 usec\nrounds: 17331"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-compile]",
            "value": 4003.96384011,
            "unit": "iter/sec",
            "range": "stddev: 0.000218357783727",
            "extra": "mean: 249.752505251 usec\nrounds: 95"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-eager]",
            "value": 309.264400519,
            "unit": "iter/sec",
            "range": "stddev: 0.0000463923841332",
            "extra": "mean: 3.23347917937 msec\nrounds: 301"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-compile]",
            "value": 5140.20303497,
            "unit": "iter/sec",
            "range": "stddev: 0.000209692708925",
            "extra": "mean: 194.544844473 usec\nrounds: 90"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-eager]",
            "value": 346.784696722,
            "unit": "iter/sec",
            "range": "stddev: 0.0000480601996731",
            "extra": "mean: 2.8836335901 msec\nrounds: 344"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-compile]",
            "value": 6270.93344437,
            "unit": "iter/sec",
            "range": "stddev: 0.00000749197818859",
            "extra": "mean: 159.465892737 usec\nrounds: 3925"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-eager]",
            "value": 18753.3840516,
            "unit": "iter/sec",
            "range": "stddev: 0.000012333453049",
            "extra": "mean: 53.3237093234 usec\nrounds: 15543"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-compile]",
            "value": 7789.67435341,
            "unit": "iter/sec",
            "range": "stddev: 0.00000518750051202",
            "extra": "mean: 128.375071233 usec\nrounds: 4576"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-eager]",
            "value": 15882.6886833,
            "unit": "iter/sec",
            "range": "stddev: 0.00244381660205",
            "extra": "mean: 62.9616319969 usec\nrounds: 18019"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-compile]",
            "value": 7437.78123898,
            "unit": "iter/sec",
            "range": "stddev: 0.0000091970366374",
            "extra": "mean: 134.448697517 usec\nrounds: 7022"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-eager]",
            "value": 22140.5703885,
            "unit": "iter/sec",
            "range": "stddev: 0.0000102604906676",
            "extra": "mean: 45.1659547361 usec\nrounds: 3005"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-compile]",
            "value": 11440.5731789,
            "unit": "iter/sec",
            "range": "stddev: 0.0000049610060509",
            "extra": "mean: 87.4082079949 usec\nrounds: 7553"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-eager]",
            "value": 42636.7661399,
            "unit": "iter/sec",
            "range": "stddev: 0.0000109026624982",
            "extra": "mean: 23.4539363684 usec\nrounds: 35550"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-compile]",
            "value": 15374.3363216,
            "unit": "iter/sec",
            "range": "stddev: 0.00000482504680545",
            "extra": "mean: 65.0434580774 usec\nrounds: 13011"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-eager]",
            "value": 45896.7279614,
            "unit": "iter/sec",
            "range": "stddev: 0.0000103481538165",
            "extra": "mean: 21.7880455626 usec\nrounds: 37398"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-compile]",
            "value": 15162.1676489,
            "unit": "iter/sec",
            "range": "stddev: 0.00000538395645776",
            "extra": "mean: 65.9536303225 usec\nrounds: 13528"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-eager]",
            "value": 44298.7829007,
            "unit": "iter/sec",
            "range": "stddev: 0.0000110053773799",
            "extra": "mean: 22.5739836294 usec\nrounds: 25050"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-compile]",
            "value": 10862.8767331,
            "unit": "iter/sec",
            "range": "stddev: 0.00000433679762183",
            "extra": "mean: 92.0566461874 usec\nrounds: 9618"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-eager]",
            "value": 45924.1295142,
            "unit": "iter/sec",
            "range": "stddev: 0.0000103926107662",
            "extra": "mean: 21.7750452884 usec\nrounds: 36325"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-compile]",
            "value": 15066.1404087,
            "unit": "iter/sec",
            "range": "stddev: 0.00000304420106385",
            "extra": "mean: 66.3739997685 usec\nrounds: 12984"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-eager]",
            "value": 44722.5248122,
            "unit": "iter/sec",
            "range": "stddev: 0.0000105211252857",
            "extra": "mean: 22.3600971591 usec\nrounds: 38536"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-compile]",
            "value": 15147.39152,
            "unit": "iter/sec",
            "range": "stddev: 0.00000309322315382",
            "extra": "mean: 66.0179674287 usec\nrounds: 13538"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-eager]",
            "value": 45270.5086901,
            "unit": "iter/sec",
            "range": "stddev: 0.000010550985743",
            "extra": "mean: 22.0894359029 usec\nrounds: 37272"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-eager]",
            "value": 32297.139062,
            "unit": "iter/sec",
            "range": "stddev: 0.00000193833534644",
            "extra": "mean: 30.9624947918 usec\nrounds: 21777"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-compile]",
            "value": 6902.15984509,
            "unit": "iter/sec",
            "range": "stddev: 0.0000069202983552",
            "extra": "mean: 144.882185061 usec\nrounds: 4285"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-eager]",
            "value": 2732.31136487,
            "unit": "iter/sec",
            "range": "stddev: 0.0000112535939534",
            "extra": "mean: 365.990499055 usec\nrounds: 2613"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-compile]",
            "value": 6165.3515484,
            "unit": "iter/sec",
            "range": "stddev: 0.00000616765373192",
            "extra": "mean: 162.196752634 usec\nrounds: 3513"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[eager]",
            "value": 6161.82426968,
            "unit": "iter/sec",
            "range": "stddev: 0.00000473682745648",
            "extra": "mean: 162.28960065 usec\nrounds: 5912"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[compile]",
            "value": 5825.54775855,
            "unit": "iter/sec",
            "range": "stddev: 0.00000672551642731",
            "extra": "mean: 171.657677775 usec\nrounds: 3780"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[eager]",
            "value": 5831.61057169,
            "unit": "iter/sec",
            "range": "stddev: 0.00000508145941451",
            "extra": "mean: 171.479214482 usec\nrounds: 5576"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[compile]",
            "value": 5365.69807733,
            "unit": "iter/sec",
            "range": "stddev: 0.0000126754761519",
            "extra": "mean: 186.369040074 usec\nrounds: 3668"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-eager]",
            "value": 109460.99749,
            "unit": "iter/sec",
            "range": "stddev: 7.77720478635e-7",
            "extra": "mean: 9.13567410248 usec\nrounds: 85245"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-compile]",
            "value": 42016.860142,
            "unit": "iter/sec",
            "range": "stddev: 0.00000266446895255",
            "extra": "mean: 23.7999697412 usec\nrounds: 26274"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-eager]",
            "value": 66357.5338115,
            "unit": "iter/sec",
            "range": "stddev: 9.51486386323e-7",
            "extra": "mean: 15.0698789205 usec\nrounds: 57176"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-compile]",
            "value": 33774.6680127,
            "unit": "iter/sec",
            "range": "stddev: 0.00000218101898448",
            "extra": "mean: 29.6079890297 usec\nrounds: 22889"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-eager]",
            "value": 37110.5385865,
            "unit": "iter/sec",
            "range": "stddev: 0.00000150938756208",
            "extra": "mean: 26.9465234968 usec\nrounds: 33156"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-compile]",
            "value": 23178.7271573,
            "unit": "iter/sec",
            "range": "stddev: 0.00000227370972136",
            "extra": "mean: 43.1430075177 usec\nrounds: 17286"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[eager]",
            "value": 38063.2332927,
            "unit": "iter/sec",
            "range": "stddev: 0.00000181044141181",
            "extra": "mean: 26.2720718524 usec\nrounds: 27042"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[compile]",
            "value": 8758.30564166,
            "unit": "iter/sec",
            "range": "stddev: 0.00000479451439801",
            "extra": "mean: 114.177335311 usec\nrounds: 7268"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[eager]",
            "value": 32664.3653662,
            "unit": "iter/sec",
            "range": "stddev: 0.00000182463486928",
            "extra": "mean: 30.6144016205 usec\nrounds: 21102"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile]",
            "value": 9368.36912137,
            "unit": "iter/sec",
            "range": "stddev: 0.000013594045469",
            "extra": "mean: 106.742164729 usec\nrounds: 5846"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile-overhead]",
            "value": 5126.47110257,
            "unit": "iter/sec",
            "range": "stddev: 0.000103465289293",
            "extra": "mean: 195.065958628 usec\nrounds: 314"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[eager]",
            "value": 3715.55687521,
            "unit": "iter/sec",
            "range": "stddev: 0.00000902237421233",
            "extra": "mean: 269.138660391 usec\nrounds: 3454"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile]",
            "value": 3203.80770001,
            "unit": "iter/sec",
            "range": "stddev: 0.00000973291170357",
            "extra": "mean: 312.128596231 usec\nrounds: 2546"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile-overhead]",
            "value": 251.505834964,
            "unit": "iter/sec",
            "range": "stddev: 0.00179536733526",
            "extra": "mean: 3.97605089418 msec\nrounds: 1030"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[False]",
            "value": 6400.63059047,
            "unit": "iter/sec",
            "range": "stddev: 0.00000546858155839",
            "extra": "mean: 156.234606242 usec\nrounds: 5295"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[True]",
            "value": 6240.99884599,
            "unit": "iter/sec",
            "range": "stddev: 0.0000112960072984",
            "extra": "mean: 160.230761882 usec\nrounds: 5426"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[eager]",
            "value": 669.367094688,
            "unit": "iter/sec",
            "range": "stddev: 0.000054936664391",
            "extra": "mean: 1.49394854921 msec\nrounds: 610"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile]",
            "value": 767.052839908,
            "unit": "iter/sec",
            "range": "stddev: 0.0000268685453583",
            "extra": "mean: 1.30369115134 msec\nrounds: 674"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile-overhead]",
            "value": 1071.07865708,
            "unit": "iter/sec",
            "range": "stddev: 0.0000776344136253",
            "extra": "mean: 933.63824719 usec\nrounds: 89"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[eager]",
            "value": 10302.806716,
            "unit": "iter/sec",
            "range": "stddev: 0.00000432756071836",
            "extra": "mean: 97.0609298575 usec\nrounds: 8753"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile]",
            "value": 8083.53873195,
            "unit": "iter/sec",
            "range": "stddev: 0.00000592761563422",
            "extra": "mean: 123.708196764 usec\nrounds: 6856"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile-overhead]",
            "value": 4906.11221626,
            "unit": "iter/sec",
            "range": "stddev: 0.00000800383579633",
            "extra": "mean: 203.827380198 usec\nrounds: 3322"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[eager]",
            "value": 2498.50173968,
            "unit": "iter/sec",
            "range": "stddev: 0.0000173597224467",
            "extra": "mean: 400.239865403 usec\nrounds: 2422"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile]",
            "value": 2971.68118674,
            "unit": "iter/sec",
            "range": "stddev: 0.00000903354178812",
            "extra": "mean: 336.509853231 usec\nrounds: 2780"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile-overhead]",
            "value": 3311.74587592,
            "unit": "iter/sec",
            "range": "stddev: 0.00003611013758",
            "extra": "mean: 301.95553568 usec\nrounds: 2270"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-eager]",
            "value": 1243.2501603,
            "unit": "iter/sec",
            "range": "stddev: 0.0000130296264672",
            "extra": "mean: 804.343350944 usec\nrounds: 1231"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile]",
            "value": 1428.74377799,
            "unit": "iter/sec",
            "range": "stddev: 0.0000163797580976",
            "extra": "mean: 699.915558971 usec\nrounds: 1365"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile-overhead]",
            "value": 2155.83155154,
            "unit": "iter/sec",
            "range": "stddev: 0.0000189970779226",
            "extra": "mean: 463.858133668 usec\nrounds: 1339"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-eager]",
            "value": 793.689546383,
            "unit": "iter/sec",
            "range": "stddev: 0.0000152746404355",
            "extra": "mean: 1.25993847917 msec\nrounds: 768"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile]",
            "value": 1379.47578295,
            "unit": "iter/sec",
            "range": "stddev: 0.0000143980653389",
            "extra": "mean: 724.913052016 usec\nrounds: 1326"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile-overhead]",
            "value": 2035.65273695,
            "unit": "iter/sec",
            "range": "stddev: 0.0000135885260726",
            "extra": "mean: 491.242922649 usec\nrounds: 1280"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-eager]",
            "value": 1244.15058172,
            "unit": "iter/sec",
            "range": "stddev: 0.0000126611095482",
            "extra": "mean: 803.761228501 usec\nrounds: 1221"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile]",
            "value": 1373.64378882,
            "unit": "iter/sec",
            "range": "stddev: 0.0000125769510001",
            "extra": "mean: 727.990770344 usec\nrounds: 1254"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile-overhead]",
            "value": 2237.84705516,
            "unit": "iter/sec",
            "range": "stddev: 0.00000948424622724",
            "extra": "mean: 446.85806284 usec\nrounds: 1114"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-eager]",
            "value": 755.022786503,
            "unit": "iter/sec",
            "range": "stddev: 0.0000222250141287",
            "extra": "mean: 1.3244633379 msec\nrounds: 728"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile]",
            "value": 1181.80821589,
            "unit": "iter/sec",
            "range": "stddev: 0.000071135399029",
            "extra": "mean: 846.160981583 usec\nrounds: 1193"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile-overhead]",
            "value": 1333.34912382,
            "unit": "iter/sec",
            "range": "stddev: 0.0000525628108368",
            "extra": "mean: 749.991117955 usec\nrounds: 975"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[eager]",
            "value": 342.183588105,
            "unit": "iter/sec",
            "range": "stddev: 0.000212701510398",
            "extra": "mean: 2.92240783826 msec\nrounds: 340"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile]",
            "value": 1285.47018465,
            "unit": "iter/sec",
            "range": "stddev: 0.0000720390417263",
            "extra": "mean: 777.925471892 usec\nrounds: 1068"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile-overhead]",
            "value": 1982.53154862,
            "unit": "iter/sec",
            "range": "stddev: 0.0000306620960267",
            "extra": "mean: 504.405592283 usec\nrounds: 130"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1066.16691862,
            "unit": "iter/sec",
            "range": "stddev: 0.0000362740573675",
            "extra": "mean: 937.939437565 usec\nrounds: 985"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1022.31617405,
            "unit": "iter/sec",
            "range": "stddev: 0.0000853222087537",
            "extra": "mean: 978.170966462 usec\nrounds: 1014"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1114.62785535,
            "unit": "iter/sec",
            "range": "stddev: 0.0000827414105067",
            "extra": "mean: 897.160424619 usec\nrounds: 869"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1165.50252694,
            "unit": "iter/sec",
            "range": "stddev: 0.0000341119096213",
            "extra": "mean: 857.998997757 usec\nrounds: 884"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-True]",
            "value": 27.819867081,
            "unit": "iter/sec",
            "range": "stddev: 0.06028218268",
            "extra": "mean: 35.9455347895 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-False]",
            "value": 38.366147317,
            "unit": "iter/sec",
            "range": "stddev: 0.000740778007036",
            "extra": "mean: 26.0646447436 msec\nrounds: 39"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-True]",
            "value": 38.5421102489,
            "unit": "iter/sec",
            "range": "stddev: 0.000181930280198",
            "extra": "mean: 25.9456473333 msec\nrounds: 39"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-False]",
            "value": 38.6250932574,
            "unit": "iter/sec",
            "range": "stddev: 0.000170885897509",
            "extra": "mean: 25.8899051282 msec\nrounds: 39"
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
          "id": "d9e4d8dcf4ee19271fc1896788d5d8a6adff34f6",
          "message": "[Feature] Group the rows of a tensordict by the value of an entry (#3098)",
          "timestamp": "2026-10-20T11:32:32-04:00",
          "tree_id": "e37e0308af0fde20c6cc4e3497c6fb094b318926",
          "url": "https://github.com/pytorch/tensordict/commit/d9e4d8dcf4ee19271fc1896788d5d8a6adff34f6"
        },
        "date": 1792511803813,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-2]",
            "value": 4556.14666104,
            "unit": "iter/sec",
            "range": "stddev: 0.00000655316056993",
            "extra": "mean: 219.483716043 usec\nrounds: 135"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-8]",
            "value": 3606.8743445,
            "unit": "iter/sec",
            "range": "stddev: 0.0000081297635491",
            "extra": "mean: 277.248360904 usec\nrounds: 3189"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-32]",
            "value": 2038.55682275,
            "unit": "iter/sec",
            "range": "stddev: 0.0000175261663175",
            "extra": "mean: 490.543108164 usec\nrounds: 1852"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-2]",
            "value": 4340.47683767,
            "unit": "iter/sec",
            "range": "stddev: 0.0000114605982599",
            "extra": "mean: 230.389433557 usec\nrounds: 3874"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-8]",
            "value": 3348.07630087,
            "unit": "iter/sec",
            "range": "stddev: 0.0000254949615487",
            "extra": "mean: 298.678975667 usec\nrounds: 2659"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-32]",
            "value": 1810.89899991,
            "unit": "iter/sec",
            "range": "stddev: 0.00000930147933366",
            "extra": "mean: 552.211912454 usec\nrounds: 1691"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-2]",
            "value": 2293.52493529,
            "unit": "iter/sec",
            "range": "stddev: 0.0000102661215638",
            "extra": "mean: 436.010084134 usec\nrounds: 72"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-8]",
            "value": 572.368129306,
            "unit": "iter/sec",
            "range": "stddev: 0.0000448766871813",
            "extra": "mean: 1.74712732733 msec\nrounds: 589"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-32]",
            "value": 144.48857624,
            "unit": "iter/sec",
            "range": "stddev: 0.0001049966355",
            "extra": "mean: 6.92096237655 msec\nrounds: 154"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-2]",
            "value": 2149.10873698,
            "unit": "iter/sec",
            "range": "stddev: 0.0000181096362",
            "extra": "mean: 465.30916877 usec\nrounds: 2146"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-8]",
            "value": 569.043970374,
            "unit": "iter/sec",
            "range": "stddev: 0.0000193558463036",
            "extra": "mean: 1.75733344357 msec\nrounds: 560"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-32]",
            "value": 144.413617363,
            "unit": "iter/sec",
            "range": "stddev: 0.000093775801678",
            "extra": "mean: 6.92455474948 msec\nrounds: 144"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-2]",
            "value": 3148.7683199,
            "unit": "iter/sec",
            "range": "stddev: 0.00000673057818266",
            "extra": "mean: 317.584496033 usec\nrounds: 2311"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-8]",
            "value": 846.499240961,
            "unit": "iter/sec",
            "range": "stddev: 0.0000131396662709",
            "extra": "mean: 1.18133596773 msec\nrounds: 825"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-32]",
            "value": 216.305424603,
            "unit": "iter/sec",
            "range": "stddev: 0.0000375481798688",
            "extra": "mean: 4.62309256383 msec\nrounds: 215"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-2]",
            "value": 3000.19421395,
            "unit": "iter/sec",
            "range": "stddev: 0.0000215319407181",
            "extra": "mean: 333.311755402 usec\nrounds: 2913"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-8]",
            "value": 836.059935405,
            "unit": "iter/sec",
            "range": "stddev: 0.0000367210256165",
            "extra": "mean: 1.19608649769 msec\nrounds: 800"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-32]",
            "value": 212.627220732,
            "unit": "iter/sec",
            "range": "stddev: 0.000142006734497",
            "extra": "mean: 4.7030666937 msec\nrounds: 212"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-2]",
            "value": 3611.9974493,
            "unit": "iter/sec",
            "range": "stddev: 0.00000866438939614",
            "extra": "mean: 276.855123526 usec\nrounds: 3004"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-8]",
            "value": 967.402421483,
            "unit": "iter/sec",
            "range": "stddev: 0.0000360798348335",
            "extra": "mean: 1.03369598607 msec\nrounds: 911"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-32]",
            "value": 249.367425294,
            "unit": "iter/sec",
            "range": "stddev: 0.0000638951675311",
            "extra": "mean: 4.01014686991 msec\nrounds: 247"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-2]",
            "value": 3650.16530982,
            "unit": "iter/sec",
            "range": "stddev: 0.00000664967093005",
            "extra": "mean: 273.960194983 usec\nrounds: 3341"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-8]",
            "value": 972.080887468,
            "unit": "iter/sec",
            "range": "stddev: 0.0000211714795427",
            "extra": "mean: 1.02872097671 msec\nrounds: 947"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-32]",
            "value": 248.521077942,
            "unit": "iter/sec",
            "range": "stddev: 0.0000203858995992",
            "extra": "mean: 4.02380356741 msec\nrounds: 245"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-128]",
            "value": 189.423650444,
            "unit": "iter/sec",
            "range": "stddev: 0.0000382816641081",
            "extra": "mean: 5.27917183338 msec\nrounds: 185"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-256]",
            "value": 95.919620768,
            "unit": "iter/sec",
            "range": "stddev: 0.00109026521024",
            "extra": "mean: 10.4253956802 msec\nrounds: 87"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-128]",
            "value": 181.816776118,
            "unit": "iter/sec",
            "range": "stddev: 0.0000960182442606",
            "extra": "mean: 5.50004252277 msec\nrounds: 168"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-256]",
            "value": 94.8322524216,
            "unit": "iter/sec",
            "range": "stddev: 0.0000461129790457",
            "extra": "mean: 10.5449356571 msec\nrounds: 88"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-128]",
            "value": 13.3575671408,
            "unit": "iter/sec",
            "range": "stddev: 0.000512860423784",
            "extra": "mean: 74.8639321412 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-256]",
            "value": 6.26354500018,
            "unit": "iter/sec",
            "range": "stddev: 0.0035507474564",
            "extra": "mean: 159.653997851 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-128]",
            "value": 12.4877792249,
            "unit": "iter/sec",
            "range": "stddev: 0.001686080368",
            "extra": "mean: 80.0782895011 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-256]",
            "value": 6.18811302412,
            "unit": "iter/sec",
            "range": "stddev: 0.00039364250401",
            "extra": "mean: 161.600151145 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-128]",
            "value": 23.7579530475,
            "unit": "iter/sec",
            "range": "stddev: 0.000342825098993",
            "extra": "mean: 42.0911682922 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-256]",
            "value": 11.8066519184,
            "unit": "iter/sec",
            "range": "stddev: 0.000680945732106",
            "extra": "mean: 84.6980165855 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-128]",
            "value": 23.6202683082,
            "unit": "iter/sec",
            "range": "stddev: 0.000136467351452",
            "extra": "mean: 42.3365216241 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-256]",
            "value": 11.8077262797,
            "unit": "iter/sec",
            "range": "stddev: 0.000560021202477",
            "extra": "mean: 84.6903100828 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-128]",
            "value": 27.638978909,
            "unit": "iter/sec",
            "range": "stddev: 0.000214505227977",
            "extra": "mean: 36.1807866814 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-256]",
            "value": 13.8315225668,
            "unit": "iter/sec",
            "range": "stddev: 0.000483754945048",
            "extra": "mean: 72.2986204284 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-128]",
            "value": 27.287585674,
            "unit": "iter/sec",
            "range": "stddev: 0.000185651783187",
            "extra": "mean: 36.646701249 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-256]",
            "value": 13.7868338887,
            "unit": "iter/sec",
            "range": "stddev: 0.000261045261915",
            "extra": "mean: 72.5329693586 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_mask_many_leaves[cuda]",
            "value": 6580.08145231,
            "unit": "iter/sec",
            "range": "stddev: 0.0000150694990581",
            "extra": "mean: 151.973802642 usec\nrounds: 41"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-None]",
            "value": 6601.3102732,
            "unit": "iter/sec",
            "range": "stddev: 0.00000629972156477",
            "extra": "mean: 151.485077752 usec\nrounds: 1429"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-cpu]",
            "value": 6565.72126984,
            "unit": "iter/sec",
            "range": "stddev: 0.00000539579191387",
            "extra": "mean: 152.306191339 usec\nrounds: 5080"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to",
            "value": 22366.0418771,
            "unit": "iter/sec",
            "range": "stddev: 0.00000274780798809",
            "extra": "mean: 44.7106379168 usec\nrounds: 11475"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_nonblocking",
            "value": 22561.4023009,
            "unit": "iter/sec",
            "range": "stddev: 0.00000294958122353",
            "extra": "mean: 44.3234860433 usec\nrounds: 11328"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[add]",
            "value": 10742.6079641,
            "unit": "iter/sec",
            "range": "stddev: 0.00000705277239114",
            "extra": "mean: 93.0872655267 usec\nrounds: 180"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[sub]",
            "value": 10645.0378373,
            "unit": "iter/sec",
            "range": "stddev: 0.00000437497377756",
            "extra": "mean: 93.9404833764 usec\nrounds: 8200"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_blocking",
            "value": 56.6842124249,
            "unit": "iter/sec",
            "range": "stddev: 0.000101430649494",
            "extra": "mean: 17.6415964379 msec\nrounds: 57"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_global_sync",
            "value": 95.7581443759,
            "unit": "iter/sec",
            "range": "stddev: 0.011926839442",
            "extra": "mean: 10.4429759632 msec\nrounds: 108"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_event_sync",
            "value": 97.655670807,
            "unit": "iter/sec",
            "range": "stddev: 0.00009044888836",
            "extra": "mean: 10.2400607331 msec\nrounds: 98"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_default",
            "value": 95.482386265,
            "unit": "iter/sec",
            "range": "stddev: 0.000121493452993",
            "extra": "mean: 10.4731358224 msec\nrounds: 96"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_cpu_to_cpu_default",
            "value": 3848.5927153,
            "unit": "iter/sec",
            "range": "stddev: 0.00000522939330313",
            "extra": "mean: 259.835236923 usec\nrounds: 3779"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[False-None]",
            "value": 295.976793164,
            "unit": "iter/sec",
            "range": "stddev: 0.0000436111098683",
            "extra": "mean: 3.37864326898 msec\nrounds: 293"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[default-None]",
            "value": 247.49432296,
            "unit": "iter/sec",
            "range": "stddev: 0.0378185495316",
            "extra": "mean: 4.04049671943 msec\nrounds: 428"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[reduce-overhead-None]",
            "value": 449.326664344,
            "unit": "iter/sec",
            "range": "stddev: 0.000271746075132",
            "extra": "mean: 2.2255523194 msec\nrounds: 432"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[False-False-None]",
            "value": 497.786629099,
            "unit": "iter/sec",
            "range": "stddev: 0.0000398922407423",
            "extra": "mean: 2.00889284995 msec\nrounds: 492"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-False-None]",
            "value": 635.819150132,
            "unit": "iter/sec",
            "range": "stddev: 0.0000649893663802",
            "extra": "mean: 1.57277427047 msec\nrounds: 640"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[within-False-None]",
            "value": 193.266292906,
            "unit": "iter/sec",
            "range": "stddev: 0.000103262133917",
            "extra": "mean: 5.17420800578 msec\nrounds: 197"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-default-None]",
            "value": 115.804702207,
            "unit": "iter/sec",
            "range": "stddev: 0.000147917069882",
            "extra": "mean: 8.63522793927 msec\nrounds: 116"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[False-False-None]",
            "value": 142.764597183,
            "unit": "iter/sec",
            "range": "stddev: 0.0000450599695857",
            "extra": "mean: 7.00453767765 msec\nrounds: 143"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[True-False-None]",
            "value": 266.509380815,
            "unit": "iter/sec",
            "range": "stddev: 0.0000452454732974",
            "extra": "mean: 3.75221313765 msec\nrounds: 266"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[within-False-None]",
            "value": 112.409941856,
            "unit": "iter/sec",
            "range": "stddev: 0.000122092745696",
            "extra": "mean: 8.89601029491 msec\nrounds: 115"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 10.8293180332,
            "unit": "iter/sec",
            "range": "stddev: 0.000829690176809",
            "extra": "mean: 92.3419182014 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_archive",
            "value": 4.51530195002,
            "unit": "iter/sec",
            "range": "stddev: 0.000899186678373",
            "extra": "mean: 221.469131205 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_zarr",
            "value": 0.442420394618,
            "unit": "iter/sec",
            "range": "stddev: 0.103037444438",
            "extra": "mean: 2.26029363059 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 41.0394777657,
            "unit": "iter/sec",
            "range": "stddev: 0.000362631649174",
            "extra": "mean: 24.3667818024 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 8.78387248595,
            "unit": "iter/sec",
            "range": "stddev: 0.0492245063088",
            "extra": "mean: 113.845004194 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 10.2756674273,
            "unit": "iter/sec",
            "range": "stddev: 0.000867139894639",
            "extra": "mean: 97.3172795901 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 43.2553700798,
            "unit": "iter/sec",
            "range": "stddev: 0.000962549668567",
            "extra": "mean: 23.1185168028 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-compile]",
            "value": 4712.44124953,
            "unit": "iter/sec",
            "range": "stddev: 0.00000928157787843",
            "extra": "mean: 212.20423705 usec\nrounds: 1783"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-eager]",
            "value": 8151.87044801,
            "unit": "iter/sec",
            "range": "stddev: 0.0000117198515924",
            "extra": "mean: 122.67123311 usec\nrounds: 7050"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-compile]",
            "value": 6590.94525993,
            "unit": "iter/sec",
            "range": "stddev: 0.0000112734376002",
            "extra": "mean: 151.723305317 usec\nrounds: 3026"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-eager]",
            "value": 5787.9662424,
            "unit": "iter/sec",
            "range": "stddev: 0.0000158683003539",
            "extra": "mean: 172.772258531 usec\nrounds: 5450"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-compile]",
            "value": 21138.5388232,
            "unit": "iter/sec",
            "range": "stddev: 0.00000304145153241",
            "extra": "mean: 47.3069595002 usec\nrounds: 16827"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-eager]",
            "value": 31288.2822244,
            "unit": "iter/sec",
            "range": "stddev: 0.00000188205828672",
            "extra": "mean: 31.9608469659 usec\nrounds: 26103"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-compile]",
            "value": 41146.8122729,
            "unit": "iter/sec",
            "range": "stddev: 0.00000353840159775",
            "extra": "mean: 24.3032192474 usec\nrounds: 38624"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-eager]",
            "value": 16045.8975051,
            "unit": "iter/sec",
            "range": "stddev: 0.000012290610581",
            "extra": "mean: 62.3212257015 usec\nrounds: 14676"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-compile]",
            "value": 3586.82417515,
            "unit": "iter/sec",
            "range": "stddev: 0.0000146875157342",
            "extra": "mean: 278.798165499 usec\nrounds: 1332"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-eager]",
            "value": 5232.91173456,
            "unit": "iter/sec",
            "range": "stddev: 0.000010242137222",
            "extra": "mean: 191.098197471 usec\nrounds: 4254"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-compile]",
            "value": 5700.07128097,
            "unit": "iter/sec",
            "range": "stddev: 0.0000120928530315",
            "extra": "mean: 175.436402583 usec\nrounds: 3130"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-eager]",
            "value": 15959.9954789,
            "unit": "iter/sec",
            "range": "stddev: 0.00000721927119077",
            "extra": "mean: 62.6566593532 usec\nrounds: 13980"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-compile]",
            "value": 4130.03706382,
            "unit": "iter/sec",
            "range": "stddev: 0.0000111237609766",
            "extra": "mean: 242.128577673 usec\nrounds: 1389"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-eager]",
            "value": 1950.83507354,
            "unit": "iter/sec",
            "range": "stddev: 0.0000198422793077",
            "extra": "mean: 512.600995114 usec\nrounds: 1892"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-eager]",
            "value": 4358.74479855,
            "unit": "iter/sec",
            "range": "stddev: 0.00000783867060398",
            "extra": "mean: 229.423847052 usec\nrounds: 4113"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-compile]",
            "value": 3749.59906701,
            "unit": "iter/sec",
            "range": "stddev: 0.00000810325980621",
            "extra": "mean: 266.695180506 usec\nrounds: 1375"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-eager]",
            "value": 12898.3920929,
            "unit": "iter/sec",
            "range": "stddev: 0.00000487780903326",
            "extra": "mean: 77.5290433722 usec\nrounds: 11379"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-compile]",
            "value": 5712.13950484,
            "unit": "iter/sec",
            "range": "stddev: 0.00000911715375661",
            "extra": "mean: 175.06575236 usec\nrounds: 2891"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-eager]",
            "value": 2212.78758298,
            "unit": "iter/sec",
            "range": "stddev: 0.0000209263294438",
            "extra": "mean: 451.918660288 usec\nrounds: 2177"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-compile]",
            "value": 4163.55833053,
            "unit": "iter/sec",
            "range": "stddev: 0.00000775666689151",
            "extra": "mean: 240.179173825 usec\nrounds: 1404"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-compile]",
            "value": 31853.244322,
            "unit": "iter/sec",
            "range": "stddev: 0.00000223096747508",
            "extra": "mean: 31.3939763838 usec\nrounds: 21954"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-eager]",
            "value": 53218.9069388,
            "unit": "iter/sec",
            "range": "stddev: 0.00000140025688582",
            "extra": "mean: 18.7903145239 usec\nrounds: 44229"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-compile]",
            "value": 33162.2416679,
            "unit": "iter/sec",
            "range": "stddev: 0.00000382188966956",
            "extra": "mean: 30.1547769302 usec\nrounds: 28369"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-eager]",
            "value": 19778.554866,
            "unit": "iter/sec",
            "range": "stddev: 0.0000117635504363",
            "extra": "mean: 50.5598112084 usec\nrounds: 16878"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-compile]",
            "value": 4024.92682516,
            "unit": "iter/sec",
            "range": "stddev: 0.000227358729238",
            "extra": "mean: 248.451721843 usec\nrounds: 94"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-eager]",
            "value": 311.777418715,
            "unit": "iter/sec",
            "range": "stddev: 0.000059061312093",
            "extra": "mean: 3.20741638096 msec\nrounds: 310"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-compile]",
            "value": 5174.89352364,
            "unit": "iter/sec",
            "range": "stddev: 0.000204005252801",
            "extra": "mean: 193.24069093 usec\nrounds: 91"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-eager]",
            "value": 349.431233173,
            "unit": "iter/sec",
            "range": "stddev: 0.000074640218049",
            "extra": "mean: 2.86179340902 msec\nrounds: 345"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-compile]",
            "value": 6361.01326291,
            "unit": "iter/sec",
            "range": "stddev: 0.00000551189832252",
            "extra": "mean: 157.207658382 usec\nrounds: 3928"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-eager]",
            "value": 18732.0691999,
            "unit": "iter/sec",
            "range": "stddev: 0.0000122760981421",
            "extra": "mean: 53.3843853195 usec\nrounds: 15342"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-compile]",
            "value": 7580.43011487,
            "unit": "iter/sec",
            "range": "stddev: 0.0000071704354741",
            "extra": "mean: 131.918635862 usec\nrounds: 4432"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-eager]",
            "value": 15517.5973643,
            "unit": "iter/sec",
            "range": "stddev: 0.00245145652744",
            "extra": "mean: 64.4429660417 usec\nrounds: 17880"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-compile]",
            "value": 7706.75935089,
            "unit": "iter/sec",
            "range": "stddev: 0.00000404712216997",
            "extra": "mean: 129.756224954 usec\nrounds: 7021"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-eager]",
            "value": 21399.4408399,
            "unit": "iter/sec",
            "range": "stddev: 0.0000102961898082",
            "extra": "mean: 46.7301929748 usec\nrounds: 19066"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-compile]",
            "value": 11332.2230354,
            "unit": "iter/sec",
            "range": "stddev: 0.00000656651356032",
            "extra": "mean: 88.2439391525 usec\nrounds: 9744"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-eager]",
            "value": 46364.4810454,
            "unit": "iter/sec",
            "range": "stddev: 0.0000104852412235",
            "extra": "mean: 21.5682345074 usec\nrounds: 36469"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-compile]",
            "value": 15504.410079,
            "unit": "iter/sec",
            "range": "stddev: 0.00000602526889632",
            "extra": "mean: 64.4977780453 usec\nrounds: 13210"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-eager]",
            "value": 47187.2379535,
            "unit": "iter/sec",
            "range": "stddev: 0.0000103800748113",
            "extra": "mean: 21.1921706667 usec\nrounds: 37763"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-compile]",
            "value": 15280.1306633,
            "unit": "iter/sec",
            "range": "stddev: 0.00000472722299786",
            "extra": "mean: 65.4444665451 usec\nrounds: 13459"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-eager]",
            "value": 42033.1228455,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113602946456",
            "extra": "mean: 23.790761483 usec\nrounds: 35460"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-compile]",
            "value": 11156.3748671,
            "unit": "iter/sec",
            "range": "stddev: 0.00000486996287325",
            "extra": "mean: 89.6348510972 usec\nrounds: 9736"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-eager]",
            "value": 41371.912664,
            "unit": "iter/sec",
            "range": "stddev: 0.0000115027756446",
            "extra": "mean: 24.1709878903 usec\nrounds: 37368"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-compile]",
            "value": 15307.2459338,
            "unit": "iter/sec",
            "range": "stddev: 0.00000411853114519",
            "extra": "mean: 65.3285381528 usec\nrounds: 13124"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-eager]",
            "value": 45691.0248227,
            "unit": "iter/sec",
            "range": "stddev: 0.0000105077859602",
            "extra": "mean: 21.8861363666 usec\nrounds: 38110"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-compile]",
            "value": 15342.0438668,
            "unit": "iter/sec",
            "range": "stddev: 0.0000045410774696",
            "extra": "mean: 65.1803637562 usec\nrounds: 13953"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-eager]",
            "value": 45533.6601022,
            "unit": "iter/sec",
            "range": "stddev: 0.0000106261176243",
            "extra": "mean: 21.9617750419 usec\nrounds: 36832"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-eager]",
            "value": 31634.3377437,
            "unit": "iter/sec",
            "range": "stddev: 0.00000255891139643",
            "extra": "mean: 31.6112196848 usec\nrounds: 22311"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-compile]",
            "value": 6689.24468039,
            "unit": "iter/sec",
            "range": "stddev: 0.0000092392495922",
            "extra": "mean: 149.493709347 usec\nrounds: 4194"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-eager]",
            "value": 2560.76504492,
            "unit": "iter/sec",
            "range": "stddev: 0.0000137401175218",
            "extra": "mean: 390.508298285 usec\nrounds: 2575"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-compile]",
            "value": 6254.22013925,
            "unit": "iter/sec",
            "range": "stddev: 0.0000095857647383",
            "extra": "mean: 159.892037334 usec\nrounds: 3325"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[eager]",
            "value": 5864.16400334,
            "unit": "iter/sec",
            "range": "stddev: 0.00000800795229539",
            "extra": "mean: 170.527290749 usec\nrounds: 5963"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[compile]",
            "value": 5760.99882636,
            "unit": "iter/sec",
            "range": "stddev: 0.00000684989526213",
            "extra": "mean: 173.581010887 usec\nrounds: 3775"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[eager]",
            "value": 5823.93912268,
            "unit": "iter/sec",
            "range": "stddev: 0.00000602651494475",
            "extra": "mean: 171.705091508 usec\nrounds: 5560"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[compile]",
            "value": 5276.07204375,
            "unit": "iter/sec",
            "range": "stddev: 0.0000073970743058",
            "extra": "mean: 189.534940332 usec\nrounds: 3498"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-eager]",
            "value": 111125.512347,
            "unit": "iter/sec",
            "range": "stddev: 9.77201555374e-7",
            "extra": "mean: 8.99883365106 usec\nrounds: 81361"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-compile]",
            "value": 41838.7455558,
            "unit": "iter/sec",
            "range": "stddev: 0.00000285008502964",
            "extra": "mean: 23.9012902207 usec\nrounds: 27225"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-eager]",
            "value": 65862.4217461,
            "unit": "iter/sec",
            "range": "stddev: 0.00000107932389955",
            "extra": "mean: 15.1831647469 usec\nrounds: 53359"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-compile]",
            "value": 33633.3181605,
            "unit": "iter/sec",
            "range": "stddev: 0.00000212782756531",
            "extra": "mean: 29.7324217381 usec\nrounds: 23736"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-eager]",
            "value": 36764.3333228,
            "unit": "iter/sec",
            "range": "stddev: 0.00000122220018759",
            "extra": "mean: 27.2002756373 usec\nrounds: 33047"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-compile]",
            "value": 24161.8684273,
            "unit": "iter/sec",
            "range": "stddev: 0.00000254545564583",
            "extra": "mean: 41.387527749 usec\nrounds: 18018"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[eager]",
            "value": 36569.9002947,
            "unit": "iter/sec",
            "range": "stddev: 0.00000201221908461",
            "extra": "mean: 27.3448927107 usec\nrounds: 29334"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[compile]",
            "value": 8538.7086968,
            "unit": "iter/sec",
            "range": "stddev: 0.00000565488024735",
            "extra": "mean: 117.113727088 usec\nrounds: 7259"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[eager]",
            "value": 30550.2336956,
            "unit": "iter/sec",
            "range": "stddev: 0.00000240136686567",
            "extra": "mean: 32.7329738281 usec\nrounds: 21877"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile]",
            "value": 9019.61511698,
            "unit": "iter/sec",
            "range": "stddev: 0.000027339296897",
            "extra": "mean: 110.869475807 usec\nrounds: 5756"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile-overhead]",
            "value": 5281.99895154,
            "unit": "iter/sec",
            "range": "stddev: 0.0000635174220461",
            "extra": "mean: 189.322264009 usec\nrounds: 314"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[eager]",
            "value": 3674.68785023,
            "unit": "iter/sec",
            "range": "stddev: 0.00000882311841945",
            "extra": "mean: 272.131958075 usec\nrounds: 3497"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile]",
            "value": 3054.31656921,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113808787199",
            "extra": "mean: 327.405485758 usec\nrounds: 2584"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile-overhead]",
            "value": 263.099509943,
            "unit": "iter/sec",
            "range": "stddev: 0.00166299921438",
            "extra": "mean: 3.80084326351 msec\nrounds: 947"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[False]",
            "value": 6063.34805648,
            "unit": "iter/sec",
            "range": "stddev: 0.00000817965088586",
            "extra": "mean: 164.925382921 usec\nrounds: 5124"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[True]",
            "value": 5991.82568171,
            "unit": "iter/sec",
            "range": "stddev: 0.00000926301921471",
            "extra": "mean: 166.894040835 usec\nrounds: 5321"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[eager]",
            "value": 636.646497795,
            "unit": "iter/sec",
            "range": "stddev: 0.0000430595685238",
            "extra": "mean: 1.57073038721 msec\nrounds: 621"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile]",
            "value": 687.458041913,
            "unit": "iter/sec",
            "range": "stddev: 0.0000450324992743",
            "extra": "mean: 1.45463423079 msec\nrounds: 631"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile-overhead]",
            "value": 1065.33502561,
            "unit": "iter/sec",
            "range": "stddev: 0.000126747946565",
            "extra": "mean: 938.671850599 usec\nrounds: 80"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[eager]",
            "value": 9960.01962493,
            "unit": "iter/sec",
            "range": "stddev: 0.00000609488775489",
            "extra": "mean: 100.401408597 usec\nrounds: 8614"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile]",
            "value": 7243.08079029,
            "unit": "iter/sec",
            "range": "stddev: 0.0000129759921158",
            "extra": "mean: 138.062797994 usec\nrounds: 6869"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile-overhead]",
            "value": 4839.83433608,
            "unit": "iter/sec",
            "range": "stddev: 0.00000681865376685",
            "extra": "mean: 206.618642408 usec\nrounds: 3260"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[eager]",
            "value": 2411.00888737,
            "unit": "iter/sec",
            "range": "stddev: 0.0000119142895649",
            "extra": "mean: 414.764128511 usec\nrounds: 2436"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile]",
            "value": 2855.4855734,
            "unit": "iter/sec",
            "range": "stddev: 0.0000219838401145",
            "extra": "mean: 350.203135087 usec\nrounds: 2684"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile-overhead]",
            "value": 3303.28100568,
            "unit": "iter/sec",
            "range": "stddev: 0.00000753294310024",
            "extra": "mean: 302.72931618 usec\nrounds: 1915"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-eager]",
            "value": 1184.95095504,
            "unit": "iter/sec",
            "range": "stddev: 0.0000206178013053",
            "extra": "mean: 843.916784697 usec\nrounds: 1239"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile]",
            "value": 1416.83748292,
            "unit": "iter/sec",
            "range": "stddev: 0.0000163908458996",
            "extra": "mean: 705.797250606 usec\nrounds: 1387"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile-overhead]",
            "value": 2140.33685147,
            "unit": "iter/sec",
            "range": "stddev: 0.0000206372023692",
            "extra": "mean: 467.216176423 usec\nrounds: 1334"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-eager]",
            "value": 792.620528557,
            "unit": "iter/sec",
            "range": "stddev: 0.0000177362281196",
            "extra": "mean: 1.26163777491 msec\nrounds: 771"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile]",
            "value": 1371.98395621,
            "unit": "iter/sec",
            "range": "stddev: 0.0000139433169932",
            "extra": "mean: 728.871496986 usec\nrounds: 1324"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile-overhead]",
            "value": 2019.21319932,
            "unit": "iter/sec",
            "range": "stddev: 0.0000144674989598",
            "extra": "mean: 495.242404484 usec\nrounds: 1204"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-eager]",
            "value": 1181.86531972,
            "unit": "iter/sec",
            "range": "stddev: 0.0000172867681346",
            "extra": "mean: 846.120097876 usec\nrounds: 1236"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile]",
            "value": 1379.56596245,
            "unit": "iter/sec",
            "range": "stddev: 0.0000257568453627",
            "extra": "mean: 724.865665882 usec\nrounds: 1390"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile-overhead]",
            "value": 2233.15800406,
            "unit": "iter/sec",
            "range": "stddev: 0.00000966395854317",
            "extra": "mean: 447.796348571 usec\nrounds: 1108"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-eager]",
            "value": 754.233560135,
            "unit": "iter/sec",
            "range": "stddev: 0.0000296207993182",
            "extra": "mean: 1.32584924996 msec\nrounds: 747"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile]",
            "value": 1216.29407108,
            "unit": "iter/sec",
            "range": "stddev: 0.0000221731502613",
            "extra": "mean: 822.169591856 usec\nrounds: 1195"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile-overhead]",
            "value": 1325.59497329,
            "unit": "iter/sec",
            "range": "stddev: 0.000115559022381",
            "extra": "mean: 754.37823781 usec\nrounds: 1031"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[eager]",
            "value": 338.693789044,
            "unit": "iter/sec",
            "range": "stddev: 0.000136820723636",
            "extra": "mean: 2.95251945075 msec\nrounds: 350"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile]",
            "value": 1318.41441803,
            "unit": "iter/sec",
            "range": "stddev: 0.0000216959166896",
            "extra": "mean: 758.486850814 usec\nrounds: 1058"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile-overhead]",
            "value": 1959.30490455,
            "unit": "iter/sec",
            "range": "stddev: 0.0000496547920818",
            "extra": "mean: 510.385084874 usec\nrounds: 119"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1065.5081346,
            "unit": "iter/sec",
            "range": "stddev: 0.0000373862944574",
            "extra": "mean: 938.519348213 usec\nrounds: 981"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1066.00357694,
            "unit": "iter/sec",
            "range": "stddev: 0.0000413314902864",
            "extra": "mean: 938.083156226 usec\nrounds: 1002"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1167.60827113,
            "unit": "iter/sec",
            "range": "stddev: 0.0000338007010457",
            "extra": "mean: 856.451623994 usec\nrounds: 1143"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1163.44902424,
            "unit": "iter/sec",
            "range": "stddev: 0.0000371022629021",
            "extra": "mean: 859.513377177 usec\nrounds: 1159"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-True]",
            "value": 27.9736213799,
            "unit": "iter/sec",
            "range": "stddev: 0.0605928842871",
            "extra": "mean: 35.7479636411 msec\nrounds: 39"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-False]",
            "value": 38.5966646988,
            "unit": "iter/sec",
            "range": "stddev: 0.000230296051228",
            "extra": "mean: 25.908974462 msec\nrounds: 39"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-True]",
            "value": 38.6141502353,
            "unit": "iter/sec",
            "range": "stddev: 0.000267820190768",
            "extra": "mean: 25.8972421744 msec\nrounds: 39"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-False]",
            "value": 38.7342006066,
            "unit": "iter/sec",
            "range": "stddev: 0.00016147456571",
            "extra": "mean: 25.8169778733 msec\nrounds: 39"
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
          "id": "17b0ddfee4ee874eac50bef247847dd93bcf3150",
          "message": "[Feature] Show tensordicts as a compact, foldable tree in notebooks (#3106)",
          "timestamp": "2026-10-20T11:54:29-04:00",
          "tree_id": "4c79f86044d5fa7985a3fee489cc7455a1c73c19",
          "url": "https://github.com/pytorch/tensordict/commit/17b0ddfee4ee874eac50bef247847dd93bcf3150"
        },
        "date": 1792513756327,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-2]",
            "value": 4366.70324459,
            "unit": "iter/sec",
            "range": "stddev: 0.0000126525875605",
            "extra": "mean: 229.005715293 usec\nrounds: 144"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-8]",
            "value": 3506.2319748,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113700114033",
            "extra": "mean: 285.206457298 usec\nrounds: 3092"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-32]",
            "value": 1988.82568947,
            "unit": "iter/sec",
            "range": "stddev: 0.0000154778418338",
            "extra": "mean: 502.809273481 usec\nrounds: 1810"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-2]",
            "value": 4383.44735435,
            "unit": "iter/sec",
            "range": "stddev: 0.0000111632012684",
            "extra": "mean: 228.130947896 usec\nrounds: 3915"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-8]",
            "value": 3402.34143486,
            "unit": "iter/sec",
            "range": "stddev: 0.0000933627202028",
            "extra": "mean: 293.91524018 usec\nrounds: 3206"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-32]",
            "value": 1767.35738659,
            "unit": "iter/sec",
            "range": "stddev: 0.0000164823264074",
            "extra": "mean: 565.816516562 usec\nrounds: 1599"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-2]",
            "value": 2168.00437783,
            "unit": "iter/sec",
            "range": "stddev: 0.0000183000850745",
            "extra": "mean: 461.25368114 usec\nrounds: 69"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-8]",
            "value": 590.785803707,
            "unit": "iter/sec",
            "range": "stddev: 0.000113644261355",
            "extra": "mean: 1.69266084887 msec\nrounds: 569"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-32]",
            "value": 151.181434256,
            "unit": "iter/sec",
            "range": "stddev: 0.000100112488086",
            "extra": "mean: 6.6145688121 msec\nrounds: 149"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-2]",
            "value": 2231.77273327,
            "unit": "iter/sec",
            "range": "stddev: 0.0000172588251347",
            "extra": "mean: 448.074297662 usec\nrounds: 2093"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-8]",
            "value": 595.68209435,
            "unit": "iter/sec",
            "range": "stddev: 0.000037673631525",
            "extra": "mean: 1.67874779095 msec\nrounds: 574"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-32]",
            "value": 153.221038219,
            "unit": "iter/sec",
            "range": "stddev: 0.0000686080155937",
            "extra": "mean: 6.52651888815 msec\nrounds: 152"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-2]",
            "value": 3062.46174187,
            "unit": "iter/sec",
            "range": "stddev: 0.0000123410748233",
            "extra": "mean: 326.534691463 usec\nrounds: 2003"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-8]",
            "value": 821.158502932,
            "unit": "iter/sec",
            "range": "stddev: 0.0000283189261966",
            "extra": "mean: 1.21779168873 msec\nrounds: 816"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-32]",
            "value": 209.964183801,
            "unit": "iter/sec",
            "range": "stddev: 0.0000565707055975",
            "extra": "mean: 4.76271705914 msec\nrounds: 203"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-2]",
            "value": 3041.57084299,
            "unit": "iter/sec",
            "range": "stddev: 0.000014391350554",
            "extra": "mean: 328.777480987 usec\nrounds: 2840"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-8]",
            "value": 821.093306009,
            "unit": "iter/sec",
            "range": "stddev: 0.0000219061987316",
            "extra": "mean: 1.21788838453 msec\nrounds: 814"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-32]",
            "value": 209.121979124,
            "unit": "iter/sec",
            "range": "stddev: 0.0000482489141755",
            "extra": "mean: 4.78189812563 msec\nrounds: 207"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-2]",
            "value": 3513.79152094,
            "unit": "iter/sec",
            "range": "stddev: 0.0000119331934366",
            "extra": "mean: 284.592866151 usec\nrounds: 2742"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-8]",
            "value": 940.164825863,
            "unit": "iter/sec",
            "range": "stddev: 0.0000294366124575",
            "extra": "mean: 1.06364328093 msec\nrounds: 922"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-32]",
            "value": 244.076694886,
            "unit": "iter/sec",
            "range": "stddev: 0.0000767685298117",
            "extra": "mean: 4.09707285027 msec\nrounds: 247"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-2]",
            "value": 3482.31399818,
            "unit": "iter/sec",
            "range": "stddev: 0.0000115310211895",
            "extra": "mean: 287.16537352 usec\nrounds: 3285"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-8]",
            "value": 943.772936619,
            "unit": "iter/sec",
            "range": "stddev: 0.0000268428156391",
            "extra": "mean: 1.05957689736 msec\nrounds: 906"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-32]",
            "value": 242.644812415,
            "unit": "iter/sec",
            "range": "stddev: 0.000077061326299",
            "extra": "mean: 4.12125027544 msec\nrounds: 236"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-128]",
            "value": 188.718452393,
            "unit": "iter/sec",
            "range": "stddev: 0.0000530056591413",
            "extra": "mean: 5.29889890108 msec\nrounds: 182"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-256]",
            "value": 98.7600884014,
            "unit": "iter/sec",
            "range": "stddev: 0.000151391022297",
            "extra": "mean: 10.1255478421 msec\nrounds: 95"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-128]",
            "value": 180.953654292,
            "unit": "iter/sec",
            "range": "stddev: 0.0000531192988341",
            "extra": "mean: 5.52627690175 msec\nrounds: 173"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-256]",
            "value": 93.7923005297,
            "unit": "iter/sec",
            "range": "stddev: 0.0000944930966988",
            "extra": "mean: 10.661855977 msec\nrounds: 87"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-128]",
            "value": 13.0249119154,
            "unit": "iter/sec",
            "range": "stddev: 0.00112815211225",
            "extra": "mean: 76.7759510768 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-256]",
            "value": 6.3245477264,
            "unit": "iter/sec",
            "range": "stddev: 0.0017701583691",
            "extra": "mean: 158.114072857 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-128]",
            "value": 12.8004441731,
            "unit": "iter/sec",
            "range": "stddev: 0.00128502104642",
            "extra": "mean: 78.1222890769 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-256]",
            "value": 6.31536486656,
            "unit": "iter/sec",
            "range": "stddev: 0.00172117407974",
            "extra": "mean: 158.343978714 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-128]",
            "value": 23.5030222717,
            "unit": "iter/sec",
            "range": "stddev: 0.00011592394627",
            "extra": "mean: 42.5477195417 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-256]",
            "value": 11.7180808907,
            "unit": "iter/sec",
            "range": "stddev: 0.000354384225949",
            "extra": "mean: 85.3382059166 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-128]",
            "value": 23.3200220866,
            "unit": "iter/sec",
            "range": "stddev: 0.000112907245376",
            "extra": "mean: 42.8816060416 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-256]",
            "value": 11.6298679505,
            "unit": "iter/sec",
            "range": "stddev: 0.000236399109788",
            "extra": "mean: 85.9854990835 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-128]",
            "value": 27.2814172634,
            "unit": "iter/sec",
            "range": "stddev: 0.000213912743591",
            "extra": "mean: 36.6549871785 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-256]",
            "value": 13.6118949162,
            "unit": "iter/sec",
            "range": "stddev: 0.00141563379592",
            "extra": "mean: 73.4651572143 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-128]",
            "value": 27.0882991328,
            "unit": "iter/sec",
            "range": "stddev: 0.000293488993218",
            "extra": "mean: 36.9163082221 msec\nrounds: 27"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-256]",
            "value": 13.5545029585,
            "unit": "iter/sec",
            "range": "stddev: 0.000525779488701",
            "extra": "mean: 73.7762205714 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_mask_many_leaves[cuda]",
            "value": 6066.32126973,
            "unit": "iter/sec",
            "range": "stddev: 0.0000176791120399",
            "extra": "mean: 164.844550022 usec\nrounds: 40"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-None]",
            "value": 6596.48901331,
            "unit": "iter/sec",
            "range": "stddev: 0.00000748012763231",
            "extra": "mean: 151.595795579 usec\nrounds: 1174"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-cpu]",
            "value": 6679.23631712,
            "unit": "iter/sec",
            "range": "stddev: 0.00000849475386768",
            "extra": "mean: 149.717715098 usec\nrounds: 5142"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to",
            "value": 22679.9200609,
            "unit": "iter/sec",
            "range": "stddev: 0.00000502896764773",
            "extra": "mean: 44.0918661668 usec\nrounds: 11701"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_nonblocking",
            "value": 23245.3480069,
            "unit": "iter/sec",
            "range": "stddev: 0.00000495116048538",
            "extra": "mean: 43.0193602481 usec\nrounds: 11925"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[add]",
            "value": 10321.4531039,
            "unit": "iter/sec",
            "range": "stddev: 0.00000893402457537",
            "extra": "mean: 96.8855828666 usec\nrounds: 175"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[sub]",
            "value": 10631.6283316,
            "unit": "iter/sec",
            "range": "stddev: 0.00000667018916346",
            "extra": "mean: 94.058969032 usec\nrounds: 7847"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_blocking",
            "value": 55.7328301878,
            "unit": "iter/sec",
            "range": "stddev: 0.000468260129021",
            "extra": "mean: 17.9427457143 msec\nrounds: 56"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_global_sync",
            "value": 89.7859761068,
            "unit": "iter/sec",
            "range": "stddev: 0.0164168594429",
            "extra": "mean: 11.1375967981 msec\nrounds: 104"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_event_sync",
            "value": 94.8430498323,
            "unit": "iter/sec",
            "range": "stddev: 0.000200755089591",
            "extra": "mean: 10.5437351685 msec\nrounds: 95"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_default",
            "value": 93.7579812398,
            "unit": "iter/sec",
            "range": "stddev: 0.000116850584125",
            "extra": "mean: 10.6657586562 msec\nrounds: 96"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_cpu_to_cpu_default",
            "value": 3837.8524579,
            "unit": "iter/sec",
            "range": "stddev: 0.00000819364360682",
            "extra": "mean: 260.562387681 usec\nrounds: 3815"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[False-None]",
            "value": 290.495341406,
            "unit": "iter/sec",
            "range": "stddev: 0.000393488179275",
            "extra": "mean: 3.44239599562 msec\nrounds: 229"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[default-None]",
            "value": 242.32851883,
            "unit": "iter/sec",
            "range": "stddev: 0.0383120531443",
            "extra": "mean: 4.12662944018 msec\nrounds: 418"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[reduce-overhead-None]",
            "value": 447.280091664,
            "unit": "iter/sec",
            "range": "stddev: 0.000248183513403",
            "extra": "mean: 2.23573554611 msec\nrounds: 423"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[False-False-None]",
            "value": 488.641138165,
            "unit": "iter/sec",
            "range": "stddev: 0.0000573438950066",
            "extra": "mean: 2.04649163137 msec\nrounds: 472"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-False-None]",
            "value": 649.688470037,
            "unit": "iter/sec",
            "range": "stddev: 0.0000405197961978",
            "extra": "mean: 1.53919924105 msec\nrounds: 614"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[within-False-None]",
            "value": 194.387019596,
            "unit": "iter/sec",
            "range": "stddev: 0.0000585172683505",
            "extra": "mean: 5.14437642019 msec\nrounds: 188"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-default-None]",
            "value": 114.289498571,
            "unit": "iter/sec",
            "range": "stddev: 0.000083867403708",
            "extra": "mean: 8.74971027525 msec\nrounds: 109"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[False-False-None]",
            "value": 138.115657086,
            "unit": "iter/sec",
            "range": "stddev: 0.0000597892078124",
            "extra": "mean: 7.24030874628 msec\nrounds: 134"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[True-False-None]",
            "value": 261.883197143,
            "unit": "iter/sec",
            "range": "stddev: 0.0000447547868464",
            "extra": "mean: 3.81849622622 msec\nrounds: 252"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[within-False-None]",
            "value": 111.649730197,
            "unit": "iter/sec",
            "range": "stddev: 0.0000645539374581",
            "extra": "mean: 8.95658232433 msec\nrounds: 111"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 15.4091715352,
            "unit": "iter/sec",
            "range": "stddev: 0.00113541481658",
            "extra": "mean: 64.8964155998 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_archive",
            "value": 4.43923238314,
            "unit": "iter/sec",
            "range": "stddev: 0.0017710379421",
            "extra": "mean: 225.2641704 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_zarr",
            "value": 0.594152221789,
            "unit": "iter/sec",
            "range": "stddev: 0.11013379282",
            "extra": "mean: 1.6830703704 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 42.2542850362,
            "unit": "iter/sec",
            "range": "stddev: 0.000700091315764",
            "extra": "mean: 23.6662388003 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 15.540715319,
            "unit": "iter/sec",
            "range": "stddev: 0.00070640251034",
            "extra": "mean: 64.3471023997 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 16.0904458119,
            "unit": "iter/sec",
            "range": "stddev: 0.000863367708254",
            "extra": "mean: 62.1486819999 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 45.7042610722,
            "unit": "iter/sec",
            "range": "stddev: 0.00029818453432",
            "extra": "mean: 21.879798 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-compile]",
            "value": 4616.26983518,
            "unit": "iter/sec",
            "range": "stddev: 0.0000098827224069",
            "extra": "mean: 216.625118484 usec\nrounds: 1688"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-eager]",
            "value": 7877.08657666,
            "unit": "iter/sec",
            "range": "stddev: 0.00000740249772448",
            "extra": "mean: 126.950489913 usec\nrounds: 6740"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-compile]",
            "value": 6179.96711709,
            "unit": "iter/sec",
            "range": "stddev: 0.00000911936099531",
            "extra": "mean: 161.81315872 usec\nrounds: 2035"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-eager]",
            "value": 5548.26738207,
            "unit": "iter/sec",
            "range": "stddev: 0.0000218973327442",
            "extra": "mean: 180.236447009 usec\nrounds: 5161"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-compile]",
            "value": 20370.027464,
            "unit": "iter/sec",
            "range": "stddev: 0.00000505605422155",
            "extra": "mean: 49.0917354807 usec\nrounds: 15753"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-eager]",
            "value": 31238.1991247,
            "unit": "iter/sec",
            "range": "stddev: 0.00000286177951619",
            "extra": "mean: 32.0120886613 usec\nrounds: 25490"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-compile]",
            "value": 58301.3855201,
            "unit": "iter/sec",
            "range": "stddev: 0.00000406063321511",
            "extra": "mean: 17.1522510328 usec\nrounds: 36832"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-eager]",
            "value": 15527.8291551,
            "unit": "iter/sec",
            "range": "stddev: 0.0000151244168909",
            "extra": "mean: 64.4005024792 usec\nrounds: 14930"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-compile]",
            "value": 3617.0440233,
            "unit": "iter/sec",
            "range": "stddev: 0.0000104351988354",
            "extra": "mean: 276.46884958 usec\nrounds: 1323"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-eager]",
            "value": 5243.90513379,
            "unit": "iter/sec",
            "range": "stddev: 0.00000970232988393",
            "extra": "mean: 190.697576422 usec\nrounds: 4835"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-compile]",
            "value": 5831.9920689,
            "unit": "iter/sec",
            "range": "stddev: 0.00000673518531812",
            "extra": "mean: 171.467997245 usec\nrounds: 3271"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-eager]",
            "value": 16131.3142897,
            "unit": "iter/sec",
            "range": "stddev: 0.00000376123386518",
            "extra": "mean: 61.9912291114 usec\nrounds: 13714"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-compile]",
            "value": 3988.03315428,
            "unit": "iter/sec",
            "range": "stddev: 0.0000170130720769",
            "extra": "mean: 250.750172155 usec\nrounds: 1365"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-eager]",
            "value": 1870.16920958,
            "unit": "iter/sec",
            "range": "stddev: 0.00002205662661",
            "extra": "mean: 534.710974213 usec\nrounds: 1783"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-eager]",
            "value": 4324.35252473,
            "unit": "iter/sec",
            "range": "stddev: 0.00000837534758123",
            "extra": "mean: 231.248491949 usec\nrounds: 4039"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-compile]",
            "value": 3658.53610765,
            "unit": "iter/sec",
            "range": "stddev: 0.00000943584091489",
            "extra": "mean: 273.333369024 usec\nrounds: 1336"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-eager]",
            "value": 12990.76109,
            "unit": "iter/sec",
            "range": "stddev: 0.00000459927507953",
            "extra": "mean: 76.9777839089 usec\nrounds: 11028"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-compile]",
            "value": 5751.05348052,
            "unit": "iter/sec",
            "range": "stddev: 0.00000803816264127",
            "extra": "mean: 173.88118601 usec\nrounds: 3274"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-eager]",
            "value": 2148.00600045,
            "unit": "iter/sec",
            "range": "stddev: 0.0000197613763362",
            "extra": "mean: 465.54804772 usec\nrounds: 2096"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-compile]",
            "value": 4102.98093702,
            "unit": "iter/sec",
            "range": "stddev: 0.00000851876884995",
            "extra": "mean: 243.725236687 usec\nrounds: 1369"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-compile]",
            "value": 36474.697881,
            "unit": "iter/sec",
            "range": "stddev: 0.00000269752474952",
            "extra": "mean: 27.4162654688 usec\nrounds: 21385"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-eager]",
            "value": 53831.034777,
            "unit": "iter/sec",
            "range": "stddev: 0.00000278265620809",
            "extra": "mean: 18.5766445721 usec\nrounds: 46445"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-compile]",
            "value": 48036.542421,
            "unit": "iter/sec",
            "range": "stddev: 0.0000118069977111",
            "extra": "mean: 20.8174849729 usec\nrounds: 26752"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-eager]",
            "value": 19576.3889905,
            "unit": "iter/sec",
            "range": "stddev: 0.0000114001077051",
            "extra": "mean: 51.0819436866 usec\nrounds: 16975"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-compile]",
            "value": 3800.34977354,
            "unit": "iter/sec",
            "range": "stddev: 0.000232547178404",
            "extra": "mean: 263.133674422 usec\nrounds: 86"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-eager]",
            "value": 310.508219691,
            "unit": "iter/sec",
            "range": "stddev: 0.0000590849170399",
            "extra": "mean: 3.22052666108 msec\nrounds: 298"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-compile]",
            "value": 4895.30473094,
            "unit": "iter/sec",
            "range": "stddev: 0.000212398836326",
            "extra": "mean: 204.277374947 usec\nrounds: 88"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-eager]",
            "value": 348.698925491,
            "unit": "iter/sec",
            "range": "stddev: 0.0000490600278131",
            "extra": "mean: 2.86780350295 msec\nrounds: 338"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-compile]",
            "value": 6155.25744477,
            "unit": "iter/sec",
            "range": "stddev: 0.00000891389033809",
            "extra": "mean: 162.462741644 usec\nrounds: 3836"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-eager]",
            "value": 18625.0666631,
            "unit": "iter/sec",
            "range": "stddev: 0.0000124809391922",
            "extra": "mean: 53.6910829954 usec\nrounds: 15013"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-compile]",
            "value": 7486.43209329,
            "unit": "iter/sec",
            "range": "stddev: 0.00000955060897087",
            "extra": "mean: 133.574977712 usec\nrounds: 4396"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-eager]",
            "value": 21327.650857,
            "unit": "iter/sec",
            "range": "stddev: 0.0000126214033786",
            "extra": "mean: 46.8874892366 usec\nrounds: 14539"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-compile]",
            "value": 7462.08803587,
            "unit": "iter/sec",
            "range": "stddev: 0.00000757323008896",
            "extra": "mean: 134.01074809 usec\nrounds: 6796"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-eager]",
            "value": 21198.5164771,
            "unit": "iter/sec",
            "range": "stddev: 0.000013049529248",
            "extra": "mean: 47.1731123771 usec\nrounds: 17708"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-compile]",
            "value": 11603.7860792,
            "unit": "iter/sec",
            "range": "stddev: 0.00000686694162584",
            "extra": "mean: 86.1787689958 usec\nrounds: 9026"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-eager]",
            "value": 46906.8410334,
            "unit": "iter/sec",
            "range": "stddev: 0.0000111505739162",
            "extra": "mean: 21.3188519621 usec\nrounds: 35612"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-compile]",
            "value": 15693.308274,
            "unit": "iter/sec",
            "range": "stddev: 0.00000923673411169",
            "extra": "mean: 63.7214271546 usec\nrounds: 12588"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-eager]",
            "value": 46254.8889882,
            "unit": "iter/sec",
            "range": "stddev: 0.0000111322074046",
            "extra": "mean: 21.619336288 usec\nrounds: 34720"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-compile]",
            "value": 15496.1943586,
            "unit": "iter/sec",
            "range": "stddev: 0.0000052160570156",
            "extra": "mean: 64.5319732614 usec\nrounds: 13240"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-eager]",
            "value": 47075.1897275,
            "unit": "iter/sec",
            "range": "stddev: 0.000010948438404",
            "extra": "mean: 21.242612208 usec\nrounds: 38255"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-compile]",
            "value": 11539.5308935,
            "unit": "iter/sec",
            "range": "stddev: 0.00000498396935309",
            "extra": "mean: 86.6586353668 usec\nrounds: 9999"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-eager]",
            "value": 47139.4725067,
            "unit": "iter/sec",
            "range": "stddev: 0.0000106383686511",
            "extra": "mean: 21.2136442523 usec\nrounds: 37791"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-compile]",
            "value": 15427.7790277,
            "unit": "iter/sec",
            "range": "stddev: 0.00000463276581191",
            "extra": "mean: 64.8181438305 usec\nrounds: 13328"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-eager]",
            "value": 46747.5316298,
            "unit": "iter/sec",
            "range": "stddev: 0.0000109077371742",
            "extra": "mean: 21.3915037893 usec\nrounds: 38240"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-compile]",
            "value": 15573.1763046,
            "unit": "iter/sec",
            "range": "stddev: 0.00000399220240055",
            "extra": "mean: 64.2129762383 usec\nrounds: 13175"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-eager]",
            "value": 46994.5559941,
            "unit": "iter/sec",
            "range": "stddev: 0.0000108997406669",
            "extra": "mean: 21.2790604964 usec\nrounds: 28497"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-eager]",
            "value": 32435.2829798,
            "unit": "iter/sec",
            "range": "stddev: 0.00000272303789924",
            "extra": "mean: 30.830623572 usec\nrounds: 22873"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-compile]",
            "value": 6786.0127678,
            "unit": "iter/sec",
            "range": "stddev: 0.0000075613126513",
            "extra": "mean: 147.361939067 usec\nrounds: 4316"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-eager]",
            "value": 2600.66329261,
            "unit": "iter/sec",
            "range": "stddev: 0.0000154681451082",
            "extra": "mean: 384.517289432 usec\nrounds: 2187"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-compile]",
            "value": 6139.68949533,
            "unit": "iter/sec",
            "range": "stddev: 0.00000674594213585",
            "extra": "mean: 162.874686214 usec\nrounds: 3429"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[eager]",
            "value": 6035.88363795,
            "unit": "iter/sec",
            "range": "stddev: 0.00000798947152694",
            "extra": "mean: 165.675824781 usec\nrounds: 5690"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[compile]",
            "value": 5502.06556817,
            "unit": "iter/sec",
            "range": "stddev: 0.00000827346125004",
            "extra": "mean: 181.749924208 usec\nrounds: 3694"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[eager]",
            "value": 5757.27061624,
            "unit": "iter/sec",
            "range": "stddev: 0.00000715593902414",
            "extra": "mean: 173.69341597 usec\nrounds: 5445"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[compile]",
            "value": 5103.13101998,
            "unit": "iter/sec",
            "range": "stddev: 0.0000125760265806",
            "extra": "mean: 195.958127684 usec\nrounds: 3524"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-eager]",
            "value": 108327.0671,
            "unit": "iter/sec",
            "range": "stddev: 0.00000162610270958",
            "extra": "mean: 9.2313031892 usec\nrounds: 82305"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-compile]",
            "value": 42640.7483677,
            "unit": "iter/sec",
            "range": "stddev: 0.00000294645806049",
            "extra": "mean: 23.4517460007 usec\nrounds: 25886"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-eager]",
            "value": 65685.4087858,
            "unit": "iter/sec",
            "range": "stddev: 0.00000141829184011",
            "extra": "mean: 15.2240812455 usec\nrounds: 52384"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-compile]",
            "value": 31576.8068502,
            "unit": "iter/sec",
            "range": "stddev: 0.00000327011545825",
            "extra": "mean: 31.6688132762 usec\nrounds: 4429"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-eager]",
            "value": 37062.0135869,
            "unit": "iter/sec",
            "range": "stddev: 0.00000289568529632",
            "extra": "mean: 26.9818043657 usec\nrounds: 33511"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-compile]",
            "value": 20201.5282108,
            "unit": "iter/sec",
            "range": "stddev: 0.00000536623218277",
            "extra": "mean: 49.5012055308 usec\nrounds: 16898"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[eager]",
            "value": 37879.0966029,
            "unit": "iter/sec",
            "range": "stddev: 0.00000270174737711",
            "extra": "mean: 26.3997848334 usec\nrounds: 27746"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[compile]",
            "value": 8219.86321148,
            "unit": "iter/sec",
            "range": "stddev: 0.00000811801635112",
            "extra": "mean: 121.656525695 usec\nrounds: 6654"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[eager]",
            "value": 30594.7345261,
            "unit": "iter/sec",
            "range": "stddev: 0.00000346624585158",
            "extra": "mean: 32.6853628733 usec\nrounds: 20255"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile]",
            "value": 8551.11898874,
            "unit": "iter/sec",
            "range": "stddev: 0.0000375830585979",
            "extra": "mean: 116.943759211 usec\nrounds: 5540"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile-overhead]",
            "value": 4759.89019781,
            "unit": "iter/sec",
            "range": "stddev: 0.000225421842796",
            "extra": "mean: 210.088879878 usec\nrounds: 308"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[eager]",
            "value": 3664.13791088,
            "unit": "iter/sec",
            "range": "stddev: 0.0000112422983908",
            "extra": "mean: 272.915491808 usec\nrounds: 3294"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile]",
            "value": 3054.92504182,
            "unit": "iter/sec",
            "range": "stddev: 0.0000108888299025",
            "extra": "mean: 327.340273922 usec\nrounds: 2362"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile-overhead]",
            "value": 260.479149309,
            "unit": "iter/sec",
            "range": "stddev: 0.00168704994281",
            "extra": "mean: 3.83907887696 msec\nrounds: 967"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[False]",
            "value": 6207.27914894,
            "unit": "iter/sec",
            "range": "stddev: 0.00000883106808458",
            "extra": "mean: 161.101180728 usec\nrounds: 5057"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[True]",
            "value": 6050.87781869,
            "unit": "iter/sec",
            "range": "stddev: 0.00000842803877119",
            "extra": "mean: 165.265277199 usec\nrounds: 5083"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[eager]",
            "value": 669.912805006,
            "unit": "iter/sec",
            "range": "stddev: 0.0000278929576261",
            "extra": "mean: 1.49273158018 msec\nrounds: 636"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile]",
            "value": 739.382371727,
            "unit": "iter/sec",
            "range": "stddev: 0.000044905555434",
            "extra": "mean: 1.35248017567 msec\nrounds: 666"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile-overhead]",
            "value": 990.85765946,
            "unit": "iter/sec",
            "range": "stddev: 0.0000877383129414",
            "extra": "mean: 1.00922669412 msec\nrounds: 85"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[eager]",
            "value": 10027.216133,
            "unit": "iter/sec",
            "range": "stddev: 0.00000749617470408",
            "extra": "mean: 99.7285773778 usec\nrounds: 8168"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile]",
            "value": 7725.04476405,
            "unit": "iter/sec",
            "range": "stddev: 0.00000716355154628",
            "extra": "mean: 129.449088069 usec\nrounds: 6313"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile-overhead]",
            "value": 4718.12997184,
            "unit": "iter/sec",
            "range": "stddev: 0.0000116266855758",
            "extra": "mean: 211.948379118 usec\nrounds: 3218"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[eager]",
            "value": 2408.34889732,
            "unit": "iter/sec",
            "range": "stddev: 0.0000157631060309",
            "extra": "mean: 415.222230099 usec\nrounds: 2286"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile]",
            "value": 2822.58451779,
            "unit": "iter/sec",
            "range": "stddev: 0.0000124797989702",
            "extra": "mean: 354.2852282 usec\nrounds: 2476"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile-overhead]",
            "value": 3229.7409137,
            "unit": "iter/sec",
            "range": "stddev: 0.0000150916079291",
            "extra": "mean: 309.622358796 usec\nrounds: 2330"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-eager]",
            "value": 1173.30175844,
            "unit": "iter/sec",
            "range": "stddev: 0.0000194628649441",
            "extra": "mean: 852.29566291 usec\nrounds: 1068"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile]",
            "value": 1405.40931274,
            "unit": "iter/sec",
            "range": "stddev: 0.0000239843100711",
            "extra": "mean: 711.536483313 usec\nrounds: 1318"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile-overhead]",
            "value": 2093.26867237,
            "unit": "iter/sec",
            "range": "stddev: 0.0000435853310172",
            "extra": "mean: 477.721762715 usec\nrounds: 1298"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-eager]",
            "value": 757.072793642,
            "unit": "iter/sec",
            "range": "stddev: 0.0000243420533938",
            "extra": "mean: 1.32087694657 msec\nrounds: 730"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile]",
            "value": 1332.99104473,
            "unit": "iter/sec",
            "range": "stddev: 0.0000173968486427",
            "extra": "mean: 750.19258678 usec\nrounds: 1273"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile-overhead]",
            "value": 1979.83900329,
            "unit": "iter/sec",
            "range": "stddev: 0.0000165579025346",
            "extra": "mean: 505.091574789 usec\nrounds: 1237"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-eager]",
            "value": 1177.01241882,
            "unit": "iter/sec",
            "range": "stddev: 0.000021145619729",
            "extra": "mean: 849.608707616 usec\nrounds: 1146"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile]",
            "value": 1392.24995096,
            "unit": "iter/sec",
            "range": "stddev: 0.0000191042823876",
            "extra": "mean: 718.261831728 usec\nrounds: 1331"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile-overhead]",
            "value": 2167.79446924,
            "unit": "iter/sec",
            "range": "stddev: 0.0000137596431996",
            "extra": "mean: 461.298344557 usec\nrounds: 1071"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-eager]",
            "value": 725.593186765,
            "unit": "iter/sec",
            "range": "stddev: 0.0000231683136173",
            "extra": "mean: 1.37818273137 msec\nrounds: 711"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile]",
            "value": 1197.47111865,
            "unit": "iter/sec",
            "range": "stddev: 0.0000206598197626",
            "extra": "mean: 835.093209706 usec\nrounds: 1135"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile-overhead]",
            "value": 1297.32526692,
            "unit": "iter/sec",
            "range": "stddev: 0.0000819166098222",
            "extra": "mean: 770.816714587 usec\nrounds: 974"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[eager]",
            "value": 337.026918297,
            "unit": "iter/sec",
            "range": "stddev: 0.0000532009313845",
            "extra": "mean: 2.96712204786 msec\nrounds: 334"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile]",
            "value": 1317.79033441,
            "unit": "iter/sec",
            "range": "stddev: 0.0000190955727354",
            "extra": "mean: 758.846057591 usec\nrounds: 1025"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile-overhead]",
            "value": 1889.14290107,
            "unit": "iter/sec",
            "range": "stddev: 0.0000308004720396",
            "extra": "mean: 529.340580552 usec\nrounds: 124"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1031.10109457,
            "unit": "iter/sec",
            "range": "stddev: 0.00003940050568",
            "extra": "mean: 969.837007508 usec\nrounds: 935"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1033.9455281,
            "unit": "iter/sec",
            "range": "stddev: 0.0000376853263983",
            "extra": "mean: 967.168939588 usec\nrounds: 1010"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1124.07662216,
            "unit": "iter/sec",
            "range": "stddev: 0.0000354048889275",
            "extra": "mean: 889.619070701 usec\nrounds: 1089"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1131.09940039,
            "unit": "iter/sec",
            "range": "stddev: 0.0000351659690556",
            "extra": "mean: 884.095597306 usec\nrounds: 1115"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-True]",
            "value": 36.8978910638,
            "unit": "iter/sec",
            "range": "stddev: 0.000155920374998",
            "extra": "mean: 27.101819946 msec\nrounds: 37"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-False]",
            "value": 36.913131626,
            "unit": "iter/sec",
            "range": "stddev: 0.000196518587525",
            "extra": "mean: 27.0906302432 msec\nrounds: 37"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-True]",
            "value": 36.8470847089,
            "unit": "iter/sec",
            "range": "stddev: 0.00060743790048",
            "extra": "mean: 27.1391891082 msec\nrounds: 37"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-False]",
            "value": 37.1009535868,
            "unit": "iter/sec",
            "range": "stddev: 0.000177114084987",
            "extra": "mean: 26.9534851081 msec\nrounds: 37"
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
          "id": "77bcc362cb70092274be2c815d6ebe8e0e896f2b",
          "message": "[BugFix] Accept jaxtyping shape annotations in tensor_only and autocast tensorclasses (#3107)",
          "timestamp": "2026-10-20T11:54:45-04:00",
          "tree_id": "8bc3271286939730f59629f638c91f1339201bac",
          "url": "https://github.com/pytorch/tensordict/commit/77bcc362cb70092274be2c815d6ebe8e0e896f2b"
        },
        "date": 1792514291720,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-2]",
            "value": 4610.25153088,
            "unit": "iter/sec",
            "range": "stddev: 0.00000630663062562",
            "extra": "mean: 216.907904764 usec\nrounds: 147"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-8]",
            "value": 3630.19186135,
            "unit": "iter/sec",
            "range": "stddev: 0.00000672848559432",
            "extra": "mean: 275.467534002 usec\nrounds: 3088"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-32]",
            "value": 2029.43374582,
            "unit": "iter/sec",
            "range": "stddev: 0.00000780140581235",
            "extra": "mean: 492.748286097 usec\nrounds: 1863"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-2]",
            "value": 4599.85224638,
            "unit": "iter/sec",
            "range": "stddev: 0.00000510567390553",
            "extra": "mean: 217.398287257 usec\nrounds: 4073"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-8]",
            "value": 3627.02329318,
            "unit": "iter/sec",
            "range": "stddev: 0.00000562895404273",
            "extra": "mean: 275.708182487 usec\nrounds: 3266"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-32]",
            "value": 1803.82105551,
            "unit": "iter/sec",
            "range": "stddev: 0.00000503534634072",
            "extra": "mean: 554.37871564 usec\nrounds: 1688"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-2]",
            "value": 2330.37523169,
            "unit": "iter/sec",
            "range": "stddev: 0.00000779149036376",
            "extra": "mean: 429.115443042 usec\nrounds: 79"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-8]",
            "value": 625.58831476,
            "unit": "iter/sec",
            "range": "stddev: 0.0000219296144109",
            "extra": "mean: 1.59849533057 msec\nrounds: 602"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-32]",
            "value": 158.455104586,
            "unit": "iter/sec",
            "range": "stddev: 0.0000554236222279",
            "extra": "mean: 6.31093584905 msec\nrounds: 159"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-2]",
            "value": 2339.31409608,
            "unit": "iter/sec",
            "range": "stddev: 0.00000722679852585",
            "extra": "mean: 427.475729606 usec\nrounds: 2182"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-8]",
            "value": 620.373092419,
            "unit": "iter/sec",
            "range": "stddev: 0.0000170313408507",
            "extra": "mean: 1.6119332257 msec\nrounds: 607"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-32]",
            "value": 157.550555693,
            "unit": "iter/sec",
            "range": "stddev: 0.0000567804796851",
            "extra": "mean: 6.34716898077 msec\nrounds: 156"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-2]",
            "value": 3168.58435023,
            "unit": "iter/sec",
            "range": "stddev: 0.00000641342076365",
            "extra": "mean: 315.598352282 usec\nrounds: 2410"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-8]",
            "value": 853.574763857,
            "unit": "iter/sec",
            "range": "stddev: 0.0000125437106562",
            "extra": "mean: 1.17154353941 msec\nrounds: 812"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-32]",
            "value": 216.616522461,
            "unit": "iter/sec",
            "range": "stddev: 0.0000291564924664",
            "extra": "mean: 4.61645302325 msec\nrounds: 215"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-2]",
            "value": 3143.77272185,
            "unit": "iter/sec",
            "range": "stddev: 0.0000063840000632",
            "extra": "mean: 318.089152263 usec\nrounds: 2962"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-8]",
            "value": 846.784004995,
            "unit": "iter/sec",
            "range": "stddev: 0.0000132570355529",
            "extra": "mean: 1.18093869759 msec\nrounds: 830"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-32]",
            "value": 215.844876233,
            "unit": "iter/sec",
            "range": "stddev: 0.0000283375495477",
            "extra": "mean: 4.63295685981 msec\nrounds: 214"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-2]",
            "value": 3689.86806418,
            "unit": "iter/sec",
            "range": "stddev: 0.00000725214891265",
            "extra": "mean: 271.012400066 usec\nrounds: 3047"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-8]",
            "value": 991.452649414,
            "unit": "iter/sec",
            "range": "stddev: 0.0000135205129004",
            "extra": "mean: 1.00862103762 msec\nrounds: 957"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-32]",
            "value": 253.659114918,
            "unit": "iter/sec",
            "range": "stddev: 0.0000354571146432",
            "extra": "mean: 3.942298704 msec\nrounds: 250"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-2]",
            "value": 3696.33793479,
            "unit": "iter/sec",
            "range": "stddev: 0.00000706540111881",
            "extra": "mean: 270.538034574 usec\nrounds: 3413"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-8]",
            "value": 992.547865906,
            "unit": "iter/sec",
            "range": "stddev: 0.0000153127859768",
            "extra": "mean: 1.00750808535 msec\nrounds: 949"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-32]",
            "value": 244.99191665,
            "unit": "iter/sec",
            "range": "stddev: 0.000285413738079",
            "extra": "mean: 4.08176732389 msec\nrounds: 247"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-128]",
            "value": 185.284147809,
            "unit": "iter/sec",
            "range": "stddev: 0.000342873391465",
            "extra": "mean: 5.39711579121 msec\nrounds: 182"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-256]",
            "value": 98.7415266939,
            "unit": "iter/sec",
            "range": "stddev: 0.0000416079026931",
            "extra": "mean: 10.1274512708 msec\nrounds: 96"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-128]",
            "value": 180.991540026,
            "unit": "iter/sec",
            "range": "stddev: 0.0000293967217757",
            "extra": "mean: 5.5251201236 msec\nrounds: 178"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-256]",
            "value": 94.6440090572,
            "unit": "iter/sec",
            "range": "stddev: 0.0000496758554117",
            "extra": "mean: 10.5659091364 msec\nrounds: 88"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-128]",
            "value": 13.4838292798,
            "unit": "iter/sec",
            "range": "stddev: 0.000662388790004",
            "extra": "mean: 74.1629087143 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-256]",
            "value": 6.7276449836,
            "unit": "iter/sec",
            "range": "stddev: 0.000643053590998",
            "extra": "mean: 148.640423571 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-128]",
            "value": 13.4247105048,
            "unit": "iter/sec",
            "range": "stddev: 0.00029143756733",
            "extra": "mean: 74.489502 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-256]",
            "value": 6.72798179556,
            "unit": "iter/sec",
            "range": "stddev: 0.000490264195091",
            "extra": "mean: 148.632982429 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-128]",
            "value": 24.0180608131,
            "unit": "iter/sec",
            "range": "stddev: 0.000132551878429",
            "extra": "mean: 41.6353346667 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-256]",
            "value": 11.9615467134,
            "unit": "iter/sec",
            "range": "stddev: 0.00019135495135",
            "extra": "mean: 83.6012285 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-128]",
            "value": 23.8038809909,
            "unit": "iter/sec",
            "range": "stddev: 0.000154161756512",
            "extra": "mean: 42.0099562917 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-256]",
            "value": 11.9138600043,
            "unit": "iter/sec",
            "range": "stddev: 0.000220000766999",
            "extra": "mean: 83.9358528333 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-128]",
            "value": 27.867569381,
            "unit": "iter/sec",
            "range": "stddev: 0.000135874705792",
            "extra": "mean: 35.8840050357 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-256]",
            "value": 13.9023966671,
            "unit": "iter/sec",
            "range": "stddev: 0.000253245422113",
            "extra": "mean: 71.9300437143 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-128]",
            "value": 27.8689211213,
            "unit": "iter/sec",
            "range": "stddev: 0.00019428144355",
            "extra": "mean: 35.8822645357 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-256]",
            "value": 13.8495625066,
            "unit": "iter/sec",
            "range": "stddev: 0.000336897105769",
            "extra": "mean: 72.2044468571 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_mask_many_leaves[cuda]",
            "value": 6472.42907015,
            "unit": "iter/sec",
            "range": "stddev: 0.0000163320561384",
            "extra": "mean: 154.501500003 usec\nrounds: 42"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-None]",
            "value": 6990.32948932,
            "unit": "iter/sec",
            "range": "stddev: 0.0000050873420636",
            "extra": "mean: 143.054773245 usec\nrounds: 1495"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-cpu]",
            "value": 6731.08219577,
            "unit": "iter/sec",
            "range": "stddev: 0.00000614431119274",
            "extra": "mean: 148.564520669 usec\nrounds: 5201"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to",
            "value": 22660.5469606,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025215723896",
            "extra": "mean: 44.1295614681 usec\nrounds: 11746"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_nonblocking",
            "value": 23649.170502,
            "unit": "iter/sec",
            "range": "stddev: 0.00000240099566029",
            "extra": "mean: 42.2847811899 usec\nrounds: 13025"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[add]",
            "value": 10763.2372422,
            "unit": "iter/sec",
            "range": "stddev: 0.00000626360072221",
            "extra": "mean: 92.9088505153 usec\nrounds: 194"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[sub]",
            "value": 11040.6112756,
            "unit": "iter/sec",
            "range": "stddev: 0.00000348318806469",
            "extra": "mean: 90.5746950998 usec\nrounds: 8265"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_blocking",
            "value": 35.3672263502,
            "unit": "iter/sec",
            "range": "stddev: 0.000114926558474",
            "extra": "mean: 28.2747646111 msec\nrounds: 36"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_global_sync",
            "value": 96.9289691469,
            "unit": "iter/sec",
            "range": "stddev: 0.0116842278728",
            "extra": "mean: 10.3168331284 msec\nrounds: 109"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_event_sync",
            "value": 98.7442271052,
            "unit": "iter/sec",
            "range": "stddev: 0.0000782889104077",
            "extra": "mean: 10.12717431 msec\nrounds: 100"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_default",
            "value": 96.6133851152,
            "unit": "iter/sec",
            "range": "stddev: 0.000100812650332",
            "extra": "mean: 10.3505326804 msec\nrounds: 97"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_cpu_to_cpu_default",
            "value": 3877.51823615,
            "unit": "iter/sec",
            "range": "stddev: 0.00000519930891564",
            "extra": "mean: 257.896917332 usec\nrounds: 3883"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[False-None]",
            "value": 297.299701189,
            "unit": "iter/sec",
            "range": "stddev: 0.0000248219850183",
            "extra": "mean: 3.3636091661 msec\nrounds: 295"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[default-None]",
            "value": 450.891467817,
            "unit": "iter/sec",
            "range": "stddev: 0.00021920432149",
            "extra": "mean: 2.21782861592 msec\nrounds: 427"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[reduce-overhead-None]",
            "value": 447.231551626,
            "unit": "iter/sec",
            "range": "stddev: 0.000292188394344",
            "extra": "mean: 2.23597820047 msec\nrounds: 424"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[False-False-None]",
            "value": 488.498459165,
            "unit": "iter/sec",
            "range": "stddev: 0.00011580676028",
            "extra": "mean: 2.04708936382 msec\nrounds: 492"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-False-None]",
            "value": 648.589233844,
            "unit": "iter/sec",
            "range": "stddev: 0.0000389435230884",
            "extra": "mean: 1.54180789291 msec\nrounds: 635"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[within-False-None]",
            "value": 195.910416908,
            "unit": "iter/sec",
            "range": "stddev: 0.0000475055783123",
            "extra": "mean: 5.10437380402 msec\nrounds: 199"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-default-None]",
            "value": 116.850538124,
            "unit": "iter/sec",
            "range": "stddev: 0.0000768436156053",
            "extra": "mean: 8.55794090517 msec\nrounds: 116"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[False-False-None]",
            "value": 141.262692795,
            "unit": "iter/sec",
            "range": "stddev: 0.000119770865322",
            "extra": "mean: 7.07900989437 msec\nrounds: 142"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[True-False-None]",
            "value": 265.518979651,
            "unit": "iter/sec",
            "range": "stddev: 0.0000532206347006",
            "extra": "mean: 3.76620910985 msec\nrounds: 264"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[within-False-None]",
            "value": 51.4213606951,
            "unit": "iter/sec",
            "range": "stddev: 0.110647354559",
            "extra": "mean: 19.4471711071 msec\nrounds: 112"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 11.2301200624,
            "unit": "iter/sec",
            "range": "stddev: 0.00379101810985",
            "extra": "mean: 89.046243 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_archive",
            "value": 4.63171488342,
            "unit": "iter/sec",
            "range": "stddev: 0.000187861381615",
            "extra": "mean: 215.9027542 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_zarr",
            "value": 0.504345746212,
            "unit": "iter/sec",
            "range": "stddev: 0.0119603917521",
            "extra": "mean: 1.9827667974 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 42.9126283953,
            "unit": "iter/sec",
            "range": "stddev: 0.000200079464705",
            "extra": "mean: 23.3031636 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.83998284836,
            "unit": "iter/sec",
            "range": "stddev: 0.0227385337923",
            "extra": "mean: 101.6261934 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 10.092641018,
            "unit": "iter/sec",
            "range": "stddev: 0.00215272255966",
            "extra": "mean: 99.0820934 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 43.9391678391,
            "unit": "iter/sec",
            "range": "stddev: 0.000684478045691",
            "extra": "mean: 22.7587378 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-compile]",
            "value": 4752.10956718,
            "unit": "iter/sec",
            "range": "stddev: 0.00000883699857908",
            "extra": "mean: 210.432858473 usec\nrounds: 1717"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-eager]",
            "value": 8149.99544661,
            "unit": "iter/sec",
            "range": "stddev: 0.00000591099068577",
            "extra": "mean: 122.699455054 usec\nrounds: 6775"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-compile]",
            "value": 6495.5748483,
            "unit": "iter/sec",
            "range": "stddev: 0.00000745540690885",
            "extra": "mean: 153.950962517 usec\nrounds: 2828"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-eager]",
            "value": 5755.02795904,
            "unit": "iter/sec",
            "range": "stddev: 0.0000167182689223",
            "extra": "mean: 173.761101965 usec\nrounds: 5345"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-compile]",
            "value": 20694.5204742,
            "unit": "iter/sec",
            "range": "stddev: 0.00000300137848211",
            "extra": "mean: 48.3219701199 usec\nrounds: 15763"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-eager]",
            "value": 31155.0232114,
            "unit": "iter/sec",
            "range": "stddev: 0.00000177961154017",
            "extra": "mean: 32.0975527193 usec\nrounds: 25465"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-compile]",
            "value": 61062.0275546,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026832706833",
            "extra": "mean: 16.3767899634 usec\nrounds: 39017"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-eager]",
            "value": 15888.5821691,
            "unit": "iter/sec",
            "range": "stddev: 0.0000128434037472",
            "extra": "mean: 62.9382779001 usec\nrounds: 14973"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-compile]",
            "value": 3656.34227678,
            "unit": "iter/sec",
            "range": "stddev: 0.00000963154032166",
            "extra": "mean: 273.497370952 usec\nrounds: 1081"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-eager]",
            "value": 5264.42180638,
            "unit": "iter/sec",
            "range": "stddev: 0.00000625725504359",
            "extra": "mean: 189.954383744 usec\nrounds: 5032"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-compile]",
            "value": 5883.65313122,
            "unit": "iter/sec",
            "range": "stddev: 0.00000560199198173",
            "extra": "mean: 169.962432811 usec\nrounds: 3066"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-eager]",
            "value": 16321.2234406,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031279189311",
            "extra": "mean: 61.2699166602 usec\nrounds: 12947"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-compile]",
            "value": 4208.09349634,
            "unit": "iter/sec",
            "range": "stddev: 0.00000710097671919",
            "extra": "mean: 237.637305556 usec\nrounds: 1368"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-eager]",
            "value": 1944.83103415,
            "unit": "iter/sec",
            "range": "stddev: 0.0000200005495438",
            "extra": "mean: 514.183485579 usec\nrounds: 1907"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-eager]",
            "value": 4357.77655262,
            "unit": "iter/sec",
            "range": "stddev: 0.00000491278818083",
            "extra": "mean: 229.474822292 usec\nrounds: 4136"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-compile]",
            "value": 3660.33319142,
            "unit": "iter/sec",
            "range": "stddev: 0.0000130809242237",
            "extra": "mean: 273.199172782 usec\nrounds: 1308"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-eager]",
            "value": 12852.3657008,
            "unit": "iter/sec",
            "range": "stddev: 0.00000454461482867",
            "extra": "mean: 77.8066873661 usec\nrounds: 11659"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-compile]",
            "value": 5596.52746579,
            "unit": "iter/sec",
            "range": "stddev: 0.00000576678754708",
            "extra": "mean: 178.682228598 usec\nrounds: 3259"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-eager]",
            "value": 2187.23454976,
            "unit": "iter/sec",
            "range": "stddev: 0.0000157305878534",
            "extra": "mean: 457.198337559 usec\nrounds: 2130"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-compile]",
            "value": 4192.71818714,
            "unit": "iter/sec",
            "range": "stddev: 0.00000610227367253",
            "extra": "mean: 238.508756221 usec\nrounds: 1407"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-compile]",
            "value": 34896.0661247,
            "unit": "iter/sec",
            "range": "stddev: 0.00000227115551415",
            "extra": "mean: 28.6565252492 usec\nrounds: 22397"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-eager]",
            "value": 55078.224821,
            "unit": "iter/sec",
            "range": "stddev: 0.00000121286823846",
            "extra": "mean: 18.1559954637 usec\nrounds: 46295"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-compile]",
            "value": 47084.2442408,
            "unit": "iter/sec",
            "range": "stddev: 0.00000145892548789",
            "extra": "mean: 21.2385271575 usec\nrounds: 29274"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-eager]",
            "value": 19863.0439775,
            "unit": "iter/sec",
            "range": "stddev: 0.0000112736462225",
            "extra": "mean: 50.3447508416 usec\nrounds: 17535"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-compile]",
            "value": 3947.63355646,
            "unit": "iter/sec",
            "range": "stddev: 0.000244810723928",
            "extra": "mean: 253.316318674 usec\nrounds: 91"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-eager]",
            "value": 301.365916001,
            "unit": "iter/sec",
            "range": "stddev: 0.000119821856629",
            "extra": "mean: 3.31822527667 msec\nrounds: 300"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-compile]",
            "value": 5067.21460912,
            "unit": "iter/sec",
            "range": "stddev: 0.000212186112227",
            "extra": "mean: 197.347078649 usec\nrounds: 89"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-eager]",
            "value": 358.762397909,
            "unit": "iter/sec",
            "range": "stddev: 0.0000442513336317",
            "extra": "mean: 2.78736011864 msec\nrounds: 354"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-compile]",
            "value": 6273.59348125,
            "unit": "iter/sec",
            "range": "stddev: 0.0000067700666046",
            "extra": "mean: 159.398278353 usec\nrounds: 3959"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-eager]",
            "value": 19205.1683687,
            "unit": "iter/sec",
            "range": "stddev: 0.0000127005288211",
            "extra": "mean: 52.069317009 usec\nrounds: 15763"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-compile]",
            "value": 7715.81228996,
            "unit": "iter/sec",
            "range": "stddev: 0.00000631221827715",
            "extra": "mean: 129.60398237 usec\nrounds: 4538"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-eager]",
            "value": 21763.7855965,
            "unit": "iter/sec",
            "range": "stddev: 0.0000119770406792",
            "extra": "mean: 45.9478887791 usec\nrounds: 18225"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-compile]",
            "value": 7696.46624262,
            "unit": "iter/sec",
            "range": "stddev: 0.00000494598951737",
            "extra": "mean: 129.929758473 usec\nrounds: 7022"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-eager]",
            "value": 22070.9898055,
            "unit": "iter/sec",
            "range": "stddev: 0.0000115824563456",
            "extra": "mean: 45.3083440667 usec\nrounds: 19037"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-compile]",
            "value": 11669.7462485,
            "unit": "iter/sec",
            "range": "stddev: 0.0000041491377308",
            "extra": "mean: 85.6916661859 usec\nrounds: 9706"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-eager]",
            "value": 46091.1884724,
            "unit": "iter/sec",
            "range": "stddev: 0.0000108286429771",
            "extra": "mean: 21.6961209538 usec\nrounds: 33922"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-compile]",
            "value": 15444.378313,
            "unit": "iter/sec",
            "range": "stddev: 0.0000082826082778",
            "extra": "mean: 64.7484786848 usec\nrounds: 13230"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-eager]",
            "value": 46137.5063777,
            "unit": "iter/sec",
            "range": "stddev: 0.0000111621616927",
            "extra": "mean: 21.6743400004 usec\nrounds: 36050"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-compile]",
            "value": 15708.0784362,
            "unit": "iter/sec",
            "range": "stddev: 0.00000302659777723",
            "extra": "mean: 63.6615104807 usec\nrounds: 13691"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-eager]",
            "value": 47811.8311223,
            "unit": "iter/sec",
            "range": "stddev: 0.0000111152140739",
            "extra": "mean: 20.9153252767 usec\nrounds: 36698"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-compile]",
            "value": 11375.4147644,
            "unit": "iter/sec",
            "range": "stddev: 0.00000380613204715",
            "extra": "mean: 87.9088825077 usec\nrounds: 10222"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-eager]",
            "value": 44836.7072123,
            "unit": "iter/sec",
            "range": "stddev: 0.000010932820223",
            "extra": "mean: 22.3031543165 usec\nrounds: 37637"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-compile]",
            "value": 15739.2189087,
            "unit": "iter/sec",
            "range": "stddev: 0.00000315180422037",
            "extra": "mean: 63.5355544514 usec\nrounds: 13682"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-eager]",
            "value": 45936.0314792,
            "unit": "iter/sec",
            "range": "stddev: 0.0000111286790976",
            "extra": "mean: 21.7694034029 usec\nrounds: 37201"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-compile]",
            "value": 15650.5025149,
            "unit": "iter/sec",
            "range": "stddev: 0.0000038647439842",
            "extra": "mean: 63.8957119138 usec\nrounds: 13673"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-eager]",
            "value": 46949.3531225,
            "unit": "iter/sec",
            "range": "stddev: 0.0000112014268069",
            "extra": "mean: 21.2995479914 usec\nrounds: 37736"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-eager]",
            "value": 33140.4123586,
            "unit": "iter/sec",
            "range": "stddev: 0.00000194834388715",
            "extra": "mean: 30.1746396266 usec\nrounds: 23792"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-compile]",
            "value": 6900.4795693,
            "unit": "iter/sec",
            "range": "stddev: 0.0000064996889891",
            "extra": "mean: 144.917464063 usec\nrounds: 4355"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-eager]",
            "value": 2631.92742593,
            "unit": "iter/sec",
            "range": "stddev: 0.0000178083774933",
            "extra": "mean: 379.949686358 usec\nrounds: 2573"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-compile]",
            "value": 6563.89381791,
            "unit": "iter/sec",
            "range": "stddev: 0.00000507700877155",
            "extra": "mean: 152.348594865 usec\nrounds: 3505"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[eager]",
            "value": 6183.14618565,
            "unit": "iter/sec",
            "range": "stddev: 0.00000452579072621",
            "extra": "mean: 161.729962381 usec\nrounds: 5981"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[compile]",
            "value": 5787.86848739,
            "unit": "iter/sec",
            "range": "stddev: 0.00000679579058372",
            "extra": "mean: 172.775176592 usec\nrounds: 3862"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[eager]",
            "value": 5806.93777794,
            "unit": "iter/sec",
            "range": "stddev: 0.00000776935114279",
            "extra": "mean: 172.207803534 usec\nrounds: 5660"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[compile]",
            "value": 5271.46443702,
            "unit": "iter/sec",
            "range": "stddev: 0.0000133544228786",
            "extra": "mean: 189.700606339 usec\nrounds: 2177"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-eager]",
            "value": 110239.891077,
            "unit": "iter/sec",
            "range": "stddev: 0.00000107199627625",
            "extra": "mean: 9.07112652447 usec\nrounds: 78309"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-compile]",
            "value": 43350.6574041,
            "unit": "iter/sec",
            "range": "stddev: 0.00000716186143734",
            "extra": "mean: 23.0677009273 usec\nrounds: 27057"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-eager]",
            "value": 67775.4114654,
            "unit": "iter/sec",
            "range": "stddev: 0.00000120389189424",
            "extra": "mean: 14.7546134856 usec\nrounds: 55646"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-compile]",
            "value": 34249.8171455,
            "unit": "iter/sec",
            "range": "stddev: 0.00000210701853591",
            "extra": "mean: 29.1972361707 usec\nrounds: 23953"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-eager]",
            "value": 36755.9788344,
            "unit": "iter/sec",
            "range": "stddev: 0.00000173545608193",
            "extra": "mean: 27.2064581522 usec\nrounds: 33299"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-compile]",
            "value": 24651.154219,
            "unit": "iter/sec",
            "range": "stddev: 0.00000374254411033",
            "extra": "mean: 40.5660518414 usec\nrounds: 18248"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[eager]",
            "value": 37564.8015189,
            "unit": "iter/sec",
            "range": "stddev: 0.00000181808995513",
            "extra": "mean: 26.6206650791 usec\nrounds: 29604"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[compile]",
            "value": 8621.76490709,
            "unit": "iter/sec",
            "range": "stddev: 0.00000480936836939",
            "extra": "mean: 115.985533215 usec\nrounds: 6759"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[eager]",
            "value": 32305.0058726,
            "unit": "iter/sec",
            "range": "stddev: 0.000003211298294",
            "extra": "mean: 30.9549549054 usec\nrounds: 22242"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile]",
            "value": 9319.76651655,
            "unit": "iter/sec",
            "range": "stddev: 0.0000303916743201",
            "extra": "mean: 107.298825376 usec\nrounds: 5927"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile-overhead]",
            "value": 5392.76763987,
            "unit": "iter/sec",
            "range": "stddev: 0.0000096320234642",
            "extra": "mean: 185.433541139 usec\nrounds: 316"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[eager]",
            "value": 3725.73695958,
            "unit": "iter/sec",
            "range": "stddev: 0.0000111704830987",
            "extra": "mean: 268.403274533 usec\nrounds: 3577"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile]",
            "value": 3170.51363579,
            "unit": "iter/sec",
            "range": "stddev: 0.0000118231962096",
            "extra": "mean: 315.406307896 usec\nrounds: 2254"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile-overhead]",
            "value": 307.349466907,
            "unit": "iter/sec",
            "range": "stddev: 0.00136388173329",
            "extra": "mean: 3.25362529521 msec\nrounds: 752"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[False]",
            "value": 6423.50847473,
            "unit": "iter/sec",
            "range": "stddev: 0.00000493724845533",
            "extra": "mean: 155.678163099 usec\nrounds: 4813"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[True]",
            "value": 6337.82259203,
            "unit": "iter/sec",
            "range": "stddev: 0.0000165656286933",
            "extra": "mean: 157.782895542 usec\nrounds: 5495"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[eager]",
            "value": 679.601596082,
            "unit": "iter/sec",
            "range": "stddev: 0.0000234168049792",
            "extra": "mean: 1.47145034056 msec\nrounds: 646"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile]",
            "value": 750.66836623,
            "unit": "iter/sec",
            "range": "stddev: 0.0000482959913827",
            "extra": "mean: 1.33214618464 msec\nrounds: 677"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile-overhead]",
            "value": 1085.37128979,
            "unit": "iter/sec",
            "range": "stddev: 0.0000776262162447",
            "extra": "mean: 921.343699993 usec\nrounds: 90"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[eager]",
            "value": 10015.8083865,
            "unit": "iter/sec",
            "range": "stddev: 0.0000055456438959",
            "extra": "mean: 99.8421656459 usec\nrounds: 8651"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile]",
            "value": 7686.98158374,
            "unit": "iter/sec",
            "range": "stddev: 0.00000931774506233",
            "extra": "mean: 130.090073601 usec\nrounds: 6970"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile-overhead]",
            "value": 4936.37719261,
            "unit": "iter/sec",
            "range": "stddev: 0.0000082811890237",
            "extra": "mean: 202.577712557 usec\nrounds: 3305"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[eager]",
            "value": 2529.20534238,
            "unit": "iter/sec",
            "range": "stddev: 0.0000150539883187",
            "extra": "mean: 395.381103797 usec\nrounds: 2476"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile]",
            "value": 2897.25835418,
            "unit": "iter/sec",
            "range": "stddev: 0.0000208924805155",
            "extra": "mean: 345.15389301 usec\nrounds: 2804"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile-overhead]",
            "value": 3214.24385529,
            "unit": "iter/sec",
            "range": "stddev: 0.00000852623597532",
            "extra": "mean: 311.115162701 usec\nrounds: 2354"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-eager]",
            "value": 1171.34343466,
            "unit": "iter/sec",
            "range": "stddev: 0.0000157380251926",
            "extra": "mean: 853.720583061 usec\nrounds: 1228"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile]",
            "value": 1377.4914675,
            "unit": "iter/sec",
            "range": "stddev: 0.0000199879864412",
            "extra": "mean: 725.957309785 usec\nrounds: 1359"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile-overhead]",
            "value": 2117.06473661,
            "unit": "iter/sec",
            "range": "stddev: 0.000044011112267",
            "extra": "mean: 472.3521122 usec\nrounds: 1123"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-eager]",
            "value": 799.907849807,
            "unit": "iter/sec",
            "range": "stddev: 0.0000268651643627",
            "extra": "mean: 1.25014400126 msec\nrounds: 792"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile]",
            "value": 1392.03906984,
            "unit": "iter/sec",
            "range": "stddev: 0.0000155758755435",
            "extra": "mean: 718.370641789 usec\nrounds: 1340"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile-overhead]",
            "value": 2041.62150895,
            "unit": "iter/sec",
            "range": "stddev: 0.0000127481447725",
            "extra": "mean: 489.806751944 usec\nrounds: 1286"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-eager]",
            "value": 1205.64777507,
            "unit": "iter/sec",
            "range": "stddev: 0.000027226120123",
            "extra": "mean: 829.429639966 usec\nrounds: 1186"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile]",
            "value": 1462.84675124,
            "unit": "iter/sec",
            "range": "stddev: 0.0000148053320758",
            "extra": "mean: 683.598606041 usec\nrounds: 1391"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile-overhead]",
            "value": 2232.34244967,
            "unit": "iter/sec",
            "range": "stddev: 0.0000125025276684",
            "extra": "mean: 447.959944564 usec\nrounds: 920"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-eager]",
            "value": 756.084612662,
            "unit": "iter/sec",
            "range": "stddev: 0.0000283478542781",
            "extra": "mean: 1.32260329499 msec\nrounds: 739"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile]",
            "value": 1200.5524656,
            "unit": "iter/sec",
            "range": "stddev: 0.0000252792215759",
            "extra": "mean: 832.949853219 usec\nrounds: 1165"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile-overhead]",
            "value": 1336.62156104,
            "unit": "iter/sec",
            "range": "stddev: 0.000086143883126",
            "extra": "mean: 748.154922191 usec\nrounds: 1041"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[eager]",
            "value": 348.856162909,
            "unit": "iter/sec",
            "range": "stddev: 0.000153346448463",
            "extra": "mean: 2.8665109186 msec\nrounds: 344"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile]",
            "value": 1347.11008853,
            "unit": "iter/sec",
            "range": "stddev: 0.0000246944237766",
            "extra": "mean: 742.329827767 usec\nrounds: 958"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile-overhead]",
            "value": 1954.30366004,
            "unit": "iter/sec",
            "range": "stddev: 0.0000718666271535",
            "extra": "mean: 511.691207692 usec\nrounds: 130"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1073.24340887,
            "unit": "iter/sec",
            "range": "stddev: 0.0000341188604629",
            "extra": "mean: 931.755081591 usec\nrounds: 956"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1066.95824835,
            "unit": "iter/sec",
            "range": "stddev: 0.0000395532378357",
            "extra": "mean: 937.24379707 usec\nrounds: 956"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1150.28138045,
            "unit": "iter/sec",
            "range": "stddev: 0.0000379067806398",
            "extra": "mean: 869.352505394 usec\nrounds: 1112"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1161.47387192,
            "unit": "iter/sec",
            "range": "stddev: 0.0000406836404458",
            "extra": "mean: 860.975028521 usec\nrounds: 1122"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-True]",
            "value": 36.7264852162,
            "unit": "iter/sec",
            "range": "stddev: 0.00165787381578",
            "extra": "mean: 27.2283066053 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-False]",
            "value": 37.8969081293,
            "unit": "iter/sec",
            "range": "stddev: 0.000450014563658",
            "extra": "mean: 26.3873769487 msec\nrounds: 39"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-True]",
            "value": 38.0621445201,
            "unit": "iter/sec",
            "range": "stddev: 0.000385724768654",
            "extra": "mean: 26.2728233684 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-False]",
            "value": 38.4184444091,
            "unit": "iter/sec",
            "range": "stddev: 0.000371871547234",
            "extra": "mean: 26.0291642564 msec\nrounds: 39"
          }
        ]
      }
    ]
  }
}