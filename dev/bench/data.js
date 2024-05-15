window.BENCHMARK_DATA = {
  "lastUpdate": 1715776468873,
  "repoUrl": "https://github.com/pytorch/tensordict",
  "entries": {
    "CPU Benchmark Results": [
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d244c13d3d3299b033297def4193872f1d1e5b87",
          "message": "[CI] Schedule workflow for release branches (#759)",
          "timestamp": "2024-04-25T14:07:23+01:00",
          "tree_id": "2e4ed25f5123da5908c75c0de26d1381d1c681c3",
          "url": "https://github.com/pytorch/tensordict/commit/d244c13d3d3299b033297def4193872f1d1e5b87"
        },
        "date": 1714050712997,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 58122.80214106356,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010583137432342754",
            "extra": "mean: 17.204951639685373 usec\nrounds: 8416"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 57865.63579491603,
            "unit": "iter/sec",
            "range": "stddev: 9.756339536493611e-7",
            "extra": "mean: 17.281413852327503 usec\nrounds: 16878"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 50597.25427162651,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026253985285379807",
            "extra": "mean: 19.763918307337306 usec\nrounds: 30627"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 51579.4103229203,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011848164763885733",
            "extra": "mean: 19.387581085928602 usec\nrounds: 33150"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 371891.4110089483,
            "unit": "iter/sec",
            "range": "stddev: 2.255668781315555e-7",
            "extra": "mean: 2.688956965386701 usec\nrounds: 107216"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 3789.4951810698535,
            "unit": "iter/sec",
            "range": "stddev: 0.000007579973065314798",
            "extra": "mean: 263.88739191315693 usec\nrounds: 3215"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 3744.4448601607533,
            "unit": "iter/sec",
            "range": "stddev: 0.00001423564612928844",
            "extra": "mean: 267.0622848902277 usec\nrounds: 3640"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 13132.807701748126,
            "unit": "iter/sec",
            "range": "stddev: 0.000002483241030183406",
            "extra": "mean: 76.14517951609758 usec\nrounds: 9754"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 3722.1724420543806,
            "unit": "iter/sec",
            "range": "stddev: 0.000006154208495373812",
            "extra": "mean: 268.66030942082557 usec\nrounds: 3471"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 12609.458014689626,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026901217751974965",
            "extra": "mean: 79.30554975757335 usec\nrounds: 10923"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 3702.134286549614,
            "unit": "iter/sec",
            "range": "stddev: 0.000007146711139610507",
            "extra": "mean: 270.11445901169594 usec\nrounds: 3623"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 190307.4669058255,
            "unit": "iter/sec",
            "range": "stddev: 3.138261019091125e-7",
            "extra": "mean: 5.254654566417273 usec\nrounds: 113689"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 7207.797003098509,
            "unit": "iter/sec",
            "range": "stddev: 0.000004016070928365393",
            "extra": "mean: 138.7386464366459 usec\nrounds: 6146"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 6963.807119360635,
            "unit": "iter/sec",
            "range": "stddev: 0.000022305042759253455",
            "extra": "mean: 143.59961194499778 usec\nrounds: 6530"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 8501.062152704459,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029562901071448996",
            "extra": "mean: 117.63235958483942 usec\nrounds: 7606"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 7226.69401583619,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032270823487084405",
            "extra": "mean: 138.37586008327645 usec\nrounds: 6461"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 8404.801135498255,
            "unit": "iter/sec",
            "range": "stddev: 0.000005732351466890784",
            "extra": "mean: 118.97961461294203 usec\nrounds: 7377"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 7033.441315981523,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037203470287331118",
            "extra": "mean: 142.17791193164297 usec\nrounds: 6529"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 860576.894359572,
            "unit": "iter/sec",
            "range": "stddev: 7.5629852465219e-8",
            "extra": "mean: 1.1620112119605355 usec\nrounds: 183487"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 19727.46079388292,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020722766534731703",
            "extra": "mean: 50.69076098785503 usec\nrounds: 15062"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 19844.57167937548,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026390378613357694",
            "extra": "mean: 50.39161419842096 usec\nrounds: 17509"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 21745.876531402922,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014732852278378319",
            "extra": "mean: 45.98572968792101 usec\nrounds: 17994"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 19030.289660555158,
            "unit": "iter/sec",
            "range": "stddev: 0.000002516554695300073",
            "extra": "mean: 52.5478076181226 usec\nrounds: 15516"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 21790.865751442525,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012878685993471007",
            "extra": "mean: 45.89078797540669 usec\nrounds: 17564"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 19073.584781770885,
            "unit": "iter/sec",
            "range": "stddev: 0.000002441164840240516",
            "extra": "mean: 52.4285293740758 usec\nrounds: 17124"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 744706.0027209134,
            "unit": "iter/sec",
            "range": "stddev: 1.671262215368931e-7",
            "extra": "mean: 1.3428117892783533 usec\nrounds: 145709"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 290462.57440643565,
            "unit": "iter/sec",
            "range": "stddev: 2.611909161770681e-7",
            "extra": "mean: 3.442784331315365 usec\nrounds: 110169"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 279022.79848175053,
            "unit": "iter/sec",
            "range": "stddev: 7.069158662246689e-7",
            "extra": "mean: 3.5839365293492493 usec\nrounds: 116605"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 271549.7497783369,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010203921515255562",
            "extra": "mean: 3.6825664572193095 usec\nrounds: 48219"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 289503.81233178155,
            "unit": "iter/sec",
            "range": "stddev: 4.0846809525515213e-7",
            "extra": "mean: 3.454185946449523 usec\nrounds: 59060"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 235532.49866387792,
            "unit": "iter/sec",
            "range": "stddev: 6.121493706989494e-7",
            "extra": "mean: 4.245698600714431 usec\nrounds: 100624"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 233958.90866416963,
            "unit": "iter/sec",
            "range": "stddev: 3.29321081534259e-7",
            "extra": "mean: 4.27425484974981 usec\nrounds: 97944"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 147738.57475596745,
            "unit": "iter/sec",
            "range": "stddev: 3.9629067324680276e-7",
            "extra": "mean: 6.768712921806551 usec\nrounds: 50798"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 145131.5544210317,
            "unit": "iter/sec",
            "range": "stddev: 4.063031733289789e-7",
            "extra": "mean: 6.890300348461542 usec\nrounds: 54813"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 93196.08396588697,
            "unit": "iter/sec",
            "range": "stddev: 8.191241433665469e-7",
            "extra": "mean: 10.730064584752672 usec\nrounds: 67214"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 98666.2423749826,
            "unit": "iter/sec",
            "range": "stddev: 5.696926743865538e-7",
            "extra": "mean: 10.135178718973448 usec\nrounds: 71190"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 94252.09696660009,
            "unit": "iter/sec",
            "range": "stddev: 6.927683992416978e-7",
            "extra": "mean: 10.609843517374133 usec\nrounds: 55054"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 98967.8912005179,
            "unit": "iter/sec",
            "range": "stddev: 4.717110983534811e-7",
            "extra": "mean: 10.104287237705302 usec\nrounds: 57764"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 88207.89133038405,
            "unit": "iter/sec",
            "range": "stddev: 9.139362955445578e-7",
            "extra": "mean: 11.336854162565617 usec\nrounds: 50234"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 96610.34949324612,
            "unit": "iter/sec",
            "range": "stddev: 7.93719374401962e-7",
            "extra": "mean: 10.350857907515472 usec\nrounds: 64268"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 89778.17932128102,
            "unit": "iter/sec",
            "range": "stddev: 8.677675024498044e-7",
            "extra": "mean: 11.13856404262099 usec\nrounds: 51005"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 98037.38518753211,
            "unit": "iter/sec",
            "range": "stddev: 5.443192354673569e-7",
            "extra": "mean: 10.200190448645042 usec\nrounds: 55301"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2814.7801029401976,
            "unit": "iter/sec",
            "range": "stddev: 0.00014455843556639375",
            "extra": "mean: 355.2675390008062 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3227.8579427713407,
            "unit": "iter/sec",
            "range": "stddev: 0.000010835810659268285",
            "extra": "mean: 309.80297699886705 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2461.4716730993123,
            "unit": "iter/sec",
            "range": "stddev: 0.001588474631379927",
            "extra": "mean: 406.26102300046796 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3125.6841243417625,
            "unit": "iter/sec",
            "range": "stddev: 0.0000090350612876755",
            "extra": "mean: 319.92996100032656 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 10468.275459782479,
            "unit": "iter/sec",
            "range": "stddev: 0.0000055168641010223165",
            "extra": "mean: 95.52671821083119 usec\nrounds: 7825"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 2406.622253639029,
            "unit": "iter/sec",
            "range": "stddev: 0.000008441570718877427",
            "extra": "mean: 415.52013345173305 usec\nrounds: 2248"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1397.414173036088,
            "unit": "iter/sec",
            "range": "stddev: 0.00006599814657894875",
            "extra": "mean: 715.6074550377236 usec\nrounds: 923"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 522225.4658163229,
            "unit": "iter/sec",
            "range": "stddev: 1.9382660562672386e-7",
            "extra": "mean: 1.914881723427329 usec\nrounds: 113818"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 97416.71803084367,
            "unit": "iter/sec",
            "range": "stddev: 6.804802302649556e-7",
            "extra": "mean: 10.26517850543255 usec\nrounds: 22201"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 75739.65682013091,
            "unit": "iter/sec",
            "range": "stddev: 7.360714256050513e-7",
            "extra": "mean: 13.203122934327965 usec\nrounds: 23720"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 61259.80729621997,
            "unit": "iter/sec",
            "range": "stddev: 8.470611555913578e-7",
            "extra": "mean: 16.323916841013386 usec\nrounds: 23425"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 73109.69681307828,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015916611094191917",
            "extra": "mean: 13.678076145722907 usec\nrounds: 14945"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 86432.01412283776,
            "unit": "iter/sec",
            "range": "stddev: 7.565256679828535e-7",
            "extra": "mean: 11.569787076566248 usec\nrounds: 11591"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 43276.3876726625,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019352130625365046",
            "extra": "mean: 23.10728907329055 usec\nrounds: 10845"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 16632.680557227548,
            "unit": "iter/sec",
            "range": "stddev: 0.000011486252350161722",
            "extra": "mean: 60.12259999579328 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 52395.89973546066,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016113631449743768",
            "extra": "mean: 19.085462890204308 usec\nrounds: 15535"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 24252.316823706657,
            "unit": "iter/sec",
            "range": "stddev: 0.000003652040547773811",
            "extra": "mean: 41.23317402082177 usec\nrounds: 7683"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 28437.194451301937,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025254337020177996",
            "extra": "mean: 35.16521299991382 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 15748.342495124283,
            "unit": "iter/sec",
            "range": "stddev: 0.000006738260616499477",
            "extra": "mean: 63.49874599879968 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 12147.389524929275,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034182434946639787",
            "extra": "mean: 82.32221399896389 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 20030.970684815686,
            "unit": "iter/sec",
            "range": "stddev: 0.000003995131088339083",
            "extra": "mean: 49.922693000496565 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 48887.63021455246,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016728103883605928",
            "extra": "mean: 20.45507208288301 usec\nrounds: 16703"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 51121.71772654606,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016665921673411446",
            "extra": "mean: 19.561158045374683 usec\nrounds: 18539"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 7112.029248707701,
            "unit": "iter/sec",
            "range": "stddev: 0.0000689430006821642",
            "extra": "mean: 140.60684581432312 usec\nrounds: 3846"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 45737.3409843912,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018873754951002337",
            "extra": "mean: 21.863973254179122 usec\nrounds: 24116"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 33381.67758243912,
            "unit": "iter/sec",
            "range": "stddev: 0.000003081015786498981",
            "extra": "mean: 29.956553187909986 usec\nrounds: 18491"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 39799.04892742132,
            "unit": "iter/sec",
            "range": "stddev: 0.000002036940953748188",
            "extra": "mean: 25.12622856449732 usec\nrounds: 12106"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 46173.31989747722,
            "unit": "iter/sec",
            "range": "stddev: 0.000001833183242283146",
            "extra": "mean: 21.65752868150677 usec\nrounds: 17555"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 38597.62529775255,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018080586591946335",
            "extra": "mean: 25.908329652037626 usec\nrounds: 15574"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 24349.155757892753,
            "unit": "iter/sec",
            "range": "stddev: 0.000010386803140933518",
            "extra": "mean: 41.069185722213426 usec\nrounds: 10968"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 16144.346514217055,
            "unit": "iter/sec",
            "range": "stddev: 0.000002212403253656462",
            "extra": "mean: 61.94118784054708 usec\nrounds: 11382"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 7999.230417748098,
            "unit": "iter/sec",
            "range": "stddev: 0.00000393517426159386",
            "extra": "mean: 125.01202587954891 usec\nrounds: 5371"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2491.493187176962,
            "unit": "iter/sec",
            "range": "stddev: 0.000008325431428065894",
            "extra": "mean: 401.36573728024956 usec\nrounds: 2162"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 889635.2883989107,
            "unit": "iter/sec",
            "range": "stddev: 7.208898125230834e-8",
            "extra": "mean: 1.1240561306866932 usec\nrounds: 175747"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3522.7325087939475,
            "unit": "iter/sec",
            "range": "stddev: 0.00006634916632930338",
            "extra": "mean: 283.8705458060347 usec\nrounds: 775"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 3889.2695861526154,
            "unit": "iter/sec",
            "range": "stddev: 0.000005934230319628036",
            "extra": "mean: 257.117686971458 usec\nrounds: 3431"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1283.3057764469872,
            "unit": "iter/sec",
            "range": "stddev: 0.0028507163744935643",
            "extra": "mean: 779.2375117087378 usec\nrounds: 1452"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 612.0568393417932,
            "unit": "iter/sec",
            "range": "stddev: 0.0025073157178079055",
            "extra": "mean: 1.6338351860840268 msec\nrounds: 618"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 609.5628221469086,
            "unit": "iter/sec",
            "range": "stddev: 0.002568651654251364",
            "extra": "mean: 1.6405199983784338 msec\nrounds: 616"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 9738.259695975796,
            "unit": "iter/sec",
            "range": "stddev: 0.000007698375319892193",
            "extra": "mean: 102.68775235201792 usec\nrounds: 2657"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 12107.766348795358,
            "unit": "iter/sec",
            "range": "stddev: 0.000037020480400836985",
            "extra": "mean: 82.59161691697936 usec\nrounds: 8784"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 180735.9897009009,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016974650725196816",
            "extra": "mean: 5.532932326621251 usec\nrounds: 22978"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1544105.834915059,
            "unit": "iter/sec",
            "range": "stddev: 1.165656058681019e-7",
            "extra": "mean: 647.6240017932513 nsec\nrounds: 74378"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 285043.94649849186,
            "unit": "iter/sec",
            "range": "stddev: 4.1098912610978824e-7",
            "extra": "mean: 3.5082309667828393 usec\nrounds: 27649"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 4224.536968957373,
            "unit": "iter/sec",
            "range": "stddev: 0.000054207779171816665",
            "extra": "mean: 236.71233258181257 usec\nrounds: 1774"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 3197.732822142697,
            "unit": "iter/sec",
            "range": "stddev: 0.000052978781792357834",
            "extra": "mean: 312.7215610621067 usec\nrounds: 2825"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1684.5165332837437,
            "unit": "iter/sec",
            "range": "stddev: 0.00005722064765230113",
            "extra": "mean: 593.6421401876248 usec\nrounds: 1498"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 8.959213809637056,
            "unit": "iter/sec",
            "range": "stddev: 0.022904617303823635",
            "extra": "mean: 111.61693662499061 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 2.6269978174738045,
            "unit": "iter/sec",
            "range": "stddev: 0.08597745653770798",
            "extra": "mean: 380.66266874999855 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.54001506790685,
            "unit": "iter/sec",
            "range": "stddev: 0.020514425010400256",
            "extra": "mean: 104.8216373749824 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 7.610960036876422,
            "unit": "iter/sec",
            "range": "stddev: 0.022763121984635412",
            "extra": "mean: 131.38946928571775 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 1.532080592207704,
            "unit": "iter/sec",
            "range": "stddev: 0.20885937410894717",
            "extra": "mean: 652.7071781250199 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_filesystem",
            "value": 10.9752148150595,
            "unit": "iter/sec",
            "range": "stddev: 0.0036588036482837274",
            "extra": "mean: 91.11438972728467 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_filesystem",
            "value": 9.970784947469763,
            "unit": "iter/sec",
            "range": "stddev: 0.021132891772740667",
            "extra": "mean: 100.29300654546412 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 38477.71686592566,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012997531558052573",
            "extra": "mean: 25.98906799705573 usec\nrounds: 9324"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 29045.187807996394,
            "unit": "iter/sec",
            "range": "stddev: 0.000001710140396290516",
            "extra": "mean: 34.42911117017089 usec\nrounds: 7511"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 39121.24818048165,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015403080511351485",
            "extra": "mean: 25.561556609508163 usec\nrounds: 13920"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 25590.832729776535,
            "unit": "iter/sec",
            "range": "stddev: 0.000001889740276093038",
            "extra": "mean: 39.07649315516167 usec\nrounds: 9642"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 34148.782135982394,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013230444924639787",
            "extra": "mean: 29.28362118502332 usec\nrounds: 10583"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 25395.430929800976,
            "unit": "iter/sec",
            "range": "stddev: 0.000005610779767325333",
            "extra": "mean: 39.37716208731556 usec\nrounds: 11173"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 34000.82796529279,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017237906081546684",
            "extra": "mean: 29.411048490371336 usec\nrounds: 8146"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 23859.462658503646,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024940490471049905",
            "extra": "mean: 41.91209225089545 usec\nrounds: 9756"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 28361.60647026199,
            "unit": "iter/sec",
            "range": "stddev: 0.000002260095171796536",
            "extra": "mean: 35.25893362382453 usec\nrounds: 8045"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 17221.142824954713,
            "unit": "iter/sec",
            "range": "stddev: 0.000003751403029712507",
            "extra": "mean: 58.06815553210127 usec\nrounds: 7529"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 9836.769204493681,
            "unit": "iter/sec",
            "range": "stddev: 0.000008760952811814644",
            "extra": "mean: 101.65939438155925 usec\nrounds: 3844"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 57109.63565309817,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018326300870917832",
            "extra": "mean: 17.51017999964688 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 28539.507541417326,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018438388695790613",
            "extra": "mean: 35.03914699820143 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 48920.96255662957,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014812249100201046",
            "extra": "mean: 20.44113500102185 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 25033.954177657113,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025983937201957634",
            "extra": "mean: 39.94574700038811 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 760.6743611762843,
            "unit": "iter/sec",
            "range": "stddev: 0.00002609324141143951",
            "extra": "mean: 1.3146229859169039 msec\nrounds: 568"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 987.1949965976792,
            "unit": "iter/sec",
            "range": "stddev: 0.00009762471790559516",
            "extra": "mean: 1.0129710983609648 msec\nrounds: 915"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6271.453715165266,
            "unit": "iter/sec",
            "range": "stddev: 0.000006737720302828436",
            "extra": "mean: 159.45266367538648 usec\nrounds: 2563"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 6700.3213724772095,
            "unit": "iter/sec",
            "range": "stddev: 0.000006511954940823852",
            "extra": "mean: 149.2465725760084 usec\nrounds: 3858"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 6837.512638290591,
            "unit": "iter/sec",
            "range": "stddev: 0.000006255263999278625",
            "extra": "mean: 146.25201486282072 usec\nrounds: 2153"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4449.147145630337,
            "unit": "iter/sec",
            "range": "stddev: 0.000018980079579395202",
            "extra": "mean: 224.7621773944103 usec\nrounds: 2610"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 2048.5823833574996,
            "unit": "iter/sec",
            "range": "stddev: 0.00005657068706597859",
            "extra": "mean: 488.14243846081604 usec\nrounds: 1560"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 2104.0703443717007,
            "unit": "iter/sec",
            "range": "stddev: 0.000020244943511047823",
            "extra": "mean: 475.2692811221629 usec\nrounds: 1960"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 2563.3392608184504,
            "unit": "iter/sec",
            "range": "stddev: 0.000021323202885835277",
            "extra": "mean: 390.1161330009471 usec\nrounds: 2406"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 2577.4713225247297,
            "unit": "iter/sec",
            "range": "stddev: 0.000018709039762529427",
            "extra": "mean: 387.97715856658397 usec\nrounds: 2428"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1820.2852057208436,
            "unit": "iter/sec",
            "range": "stddev: 0.000022999216453178116",
            "extra": "mean: 549.3644605016685 usec\nrounds: 1633"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1818.2951631840515,
            "unit": "iter/sec",
            "range": "stddev: 0.000019393971160135636",
            "extra": "mean: 549.9657152741257 usec\nrounds: 1735"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 2203.235831062831,
            "unit": "iter/sec",
            "range": "stddev: 0.000026857006376004377",
            "extra": "mean: 453.8778763041469 usec\nrounds: 2013"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 2214.9086994080494,
            "unit": "iter/sec",
            "range": "stddev: 0.000015408237041566523",
            "extra": "mean: 451.48587852278393 usec\nrounds: 2058"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 591.957875643757,
            "unit": "iter/sec",
            "range": "stddev: 0.00008489300264965789",
            "extra": "mean: 1.689309393700515 msec\nrounds: 508"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 600.9089263684908,
            "unit": "iter/sec",
            "range": "stddev: 0.00008595052187266353",
            "extra": "mean: 1.6641456901686587 msec\nrounds: 539"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c9f714afd4ce1e326c5ad5744380851f6b3974f1",
          "message": "[Doc] Per-release docs (#758)",
          "timestamp": "2024-04-25T14:02:11+01:00",
          "tree_id": "99b9beea3ac0aea2c31bcfc69ac1bc1b7ab82308",
          "url": "https://github.com/pytorch/tensordict/commit/c9f714afd4ce1e326c5ad5744380851f6b3974f1"
        },
        "date": 1714051063717,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 59873.248952379756,
            "unit": "iter/sec",
            "range": "stddev: 0.000001029382397221962",
            "extra": "mean: 16.701949827298513 usec\nrounds: 8092"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 58577.28860470106,
            "unit": "iter/sec",
            "range": "stddev: 0.00000117235432783403",
            "extra": "mean: 17.071462742981005 usec\nrounds: 16990"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 51616.998080311,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012456116976236877",
            "extra": "mean: 19.373462951954274 usec\nrounds: 30366"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 51659.10138789781,
            "unit": "iter/sec",
            "range": "stddev: 0.000001397450595282068",
            "extra": "mean: 19.357673152136364 usec\nrounds: 32177"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 368574.75315783295,
            "unit": "iter/sec",
            "range": "stddev: 2.4525656681724116e-7",
            "extra": "mean: 2.7131538213952893 usec\nrounds: 98146"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 3757.2414696963438,
            "unit": "iter/sec",
            "range": "stddev: 0.000012226123403212719",
            "extra": "mean: 266.1527101905481 usec\nrounds: 2091"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 3719.2964747662218,
            "unit": "iter/sec",
            "range": "stddev: 0.000005854438965485598",
            "extra": "mean: 268.8680525428819 usec\nrounds: 3559"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 12744.565953416963,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026665886469717625",
            "extra": "mean: 78.46481423181687 usec\nrounds: 9921"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 3746.245044006155,
            "unit": "iter/sec",
            "range": "stddev: 0.000007002941673632857",
            "extra": "mean: 266.93395339954094 usec\nrounds: 3369"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 12354.031414416697,
            "unit": "iter/sec",
            "range": "stddev: 0.000006640579348716617",
            "extra": "mean: 80.94523693966303 usec\nrounds: 10893"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 3721.329065950251,
            "unit": "iter/sec",
            "range": "stddev: 0.0000044699385326832775",
            "extra": "mean: 268.7211967223994 usec\nrounds: 3599"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 236701.99855154764,
            "unit": "iter/sec",
            "range": "stddev: 4.3265757607881955e-7",
            "extra": "mean: 4.224721405477384 usec\nrounds: 87177"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 7241.65407419144,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034800704321077725",
            "extra": "mean: 138.08999846649732 usec\nrounds: 5870"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 6951.663326898098,
            "unit": "iter/sec",
            "range": "stddev: 0.00002238982842856006",
            "extra": "mean: 143.85046469823936 usec\nrounds: 6430"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 8501.194952776508,
            "unit": "iter/sec",
            "range": "stddev: 0.000003643286606627754",
            "extra": "mean: 117.63052200954385 usec\nrounds: 7247"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 7168.997201464363,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033109269144224174",
            "extra": "mean: 139.4895229971267 usec\nrounds: 6457"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 8492.459410882078,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037709677919527466",
            "extra": "mean: 117.7515195090151 usec\nrounds: 7228"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 6916.3234808503275,
            "unit": "iter/sec",
            "range": "stddev: 0.000008410749578373778",
            "extra": "mean: 144.5854871838723 usec\nrounds: 6398"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 717512.193404133,
            "unit": "iter/sec",
            "range": "stddev: 1.8327781878215516e-7",
            "extra": "mean: 1.393704537975368 usec\nrounds: 187618"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 19777.796583043375,
            "unit": "iter/sec",
            "range": "stddev: 0.000001938052738945418",
            "extra": "mean: 50.561749677279856 usec\nrounds: 14713"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 19795.90542568714,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016985848428894985",
            "extra": "mean: 50.51549694223137 usec\nrounds: 17660"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 21985.522480919044,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015984609663840827",
            "extra": "mean: 45.48447738132615 usec\nrounds: 18214"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 19416.685416508673,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017346729731959185",
            "extra": "mean: 51.50209618938197 usec\nrounds: 15511"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 21940.648290005687,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015249405235543635",
            "extra": "mean: 45.57750467453215 usec\nrounds: 17542"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 19359.30662073842,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017605348627254174",
            "extra": "mean: 51.65474257888773 usec\nrounds: 17617"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 759645.7716687217,
            "unit": "iter/sec",
            "range": "stddev: 1.4139039713384307e-7",
            "extra": "mean: 1.3164030358561591 usec\nrounds: 127470"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 292598.4587755982,
            "unit": "iter/sec",
            "range": "stddev: 2.9180683864487364e-7",
            "extra": "mean: 3.417652998531094 usec\nrounds: 107331"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 289825.39865942043,
            "unit": "iter/sec",
            "range": "stddev: 3.5279886988708046e-7",
            "extra": "mean: 3.4503532286178955 usec\nrounds: 104516"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 293507.5476907435,
            "unit": "iter/sec",
            "range": "stddev: 3.146771684939697e-7",
            "extra": "mean: 3.4070674088887745 usec\nrounds: 73714"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 294213.8893249901,
            "unit": "iter/sec",
            "range": "stddev: 2.9374618409130986e-7",
            "extra": "mean: 3.398887803340226 usec\nrounds: 44048"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 236574.75663033608,
            "unit": "iter/sec",
            "range": "stddev: 4.1542287124330543e-7",
            "extra": "mean: 4.226993675248991 usec\nrounds: 101031"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 236613.85077092808,
            "unit": "iter/sec",
            "range": "stddev: 4.009352890629775e-7",
            "extra": "mean: 4.226295277059354 usec\nrounds: 101647"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 213302.98506762463,
            "unit": "iter/sec",
            "range": "stddev: 0.000001841694637526677",
            "extra": "mean: 4.68816692688555 usec\nrounds: 59553"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 238482.84024878786,
            "unit": "iter/sec",
            "range": "stddev: 4.159884704315657e-7",
            "extra": "mean: 4.193173810563431 usec\nrounds: 75444"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 91913.22780088683,
            "unit": "iter/sec",
            "range": "stddev: 8.325149498904626e-7",
            "extra": "mean: 10.879826809763626 usec\nrounds: 67440"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 98258.35533595791,
            "unit": "iter/sec",
            "range": "stddev: 6.699580784803411e-7",
            "extra": "mean: 10.177251558718563 usec\nrounds: 67857"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 91544.8231152493,
            "unit": "iter/sec",
            "range": "stddev: 9.049606789810455e-7",
            "extra": "mean: 10.923610598286498 usec\nrounds: 51081"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 97909.0475826297,
            "unit": "iter/sec",
            "range": "stddev: 5.579927240992636e-7",
            "extra": "mean: 10.213560694235703 usec\nrounds: 60129"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 87134.11738981152,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012310124795069077",
            "extra": "mean: 11.476560846152884 usec\nrounds: 50718"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 95498.58365213014,
            "unit": "iter/sec",
            "range": "stddev: 5.380639283832464e-7",
            "extra": "mean: 10.471359487829373 usec\nrounds: 65151"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 89061.17518035827,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015310548969283302",
            "extra": "mean: 11.22823719735221 usec\nrounds: 49073"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 96295.09674660499,
            "unit": "iter/sec",
            "range": "stddev: 5.961522369483259e-7",
            "extra": "mean: 10.38474474594945 usec\nrounds: 55854"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2797.0959520491588,
            "unit": "iter/sec",
            "range": "stddev: 0.0001568272593880252",
            "extra": "mean: 357.5136560000374 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3207.920339916208,
            "unit": "iter/sec",
            "range": "stddev: 0.00001713337394921827",
            "extra": "mean: 311.72843900048974 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2392.36555385641,
            "unit": "iter/sec",
            "range": "stddev: 0.0019102085876956212",
            "extra": "mean: 417.9963210003734 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3106.431345510135,
            "unit": "iter/sec",
            "range": "stddev: 0.000015069135920518973",
            "extra": "mean: 321.9127959951038 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 10273.899671992174,
            "unit": "iter/sec",
            "range": "stddev: 0.000008879261059697112",
            "extra": "mean: 97.33402426793346 usec\nrounds: 6758"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 2366.0702213260925,
            "unit": "iter/sec",
            "range": "stddev: 0.000025886047833002972",
            "extra": "mean: 422.64172507929123 usec\nrounds: 2197"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1377.1369321301497,
            "unit": "iter/sec",
            "range": "stddev: 0.00017480226427027192",
            "extra": "mean: 726.1442029974493 usec\nrounds: 867"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 522681.84251129575,
            "unit": "iter/sec",
            "range": "stddev: 2.4372548207091587e-7",
            "extra": "mean: 1.9132097552793579 usec\nrounds: 96722"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 101215.1225212298,
            "unit": "iter/sec",
            "range": "stddev: 6.388340333142303e-7",
            "extra": "mean: 9.879946544452888 usec\nrounds: 20185"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 78513.8877196087,
            "unit": "iter/sec",
            "range": "stddev: 8.683356965223394e-7",
            "extra": "mean: 12.736600225061226 usec\nrounds: 22205"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 62712.05372993749,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015835613903448194",
            "extra": "mean: 15.945897806287594 usec\nrounds: 20383"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 70264.88521866094,
            "unit": "iter/sec",
            "range": "stddev: 0.000002729965320077878",
            "extra": "mean: 14.231859866959834 usec\nrounds: 13266"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 86506.70182078099,
            "unit": "iter/sec",
            "range": "stddev: 7.240299350132937e-7",
            "extra": "mean: 11.559798015091772 usec\nrounds: 11189"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 42151.130412288425,
            "unit": "iter/sec",
            "range": "stddev: 0.000002288137472531388",
            "extra": "mean: 23.7241561547414 usec\nrounds: 10368"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 15971.532348025472,
            "unit": "iter/sec",
            "range": "stddev: 0.000016071700563133867",
            "extra": "mean: 62.61139997150167 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 52443.48000595755,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014915129514211483",
            "extra": "mean: 19.06814726799977 usec\nrounds: 14898"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 24130.416471772653,
            "unit": "iter/sec",
            "range": "stddev: 0.0000048488960112285035",
            "extra": "mean: 41.4414728883682 usec\nrounds: 7211"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 28234.669632772453,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028681436188971508",
            "extra": "mean: 35.41745000052288 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 15889.549327793085,
            "unit": "iter/sec",
            "range": "stddev: 0.000006794681773320705",
            "extra": "mean: 62.93444699849715 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 11623.18472022326,
            "unit": "iter/sec",
            "range": "stddev: 0.000005598607397988817",
            "extra": "mean: 86.03493999885359 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 19921.609660387912,
            "unit": "iter/sec",
            "range": "stddev: 0.00000430436463424271",
            "extra": "mean: 50.19674700224641 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 47790.705265063036,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020917955553910507",
            "extra": "mean: 20.924570885775168 usec\nrounds: 14742"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 49158.012010799575,
            "unit": "iter/sec",
            "range": "stddev: 0.000003028717643528313",
            "extra": "mean: 20.342563889286428 usec\nrounds: 17812"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 6873.106844794542,
            "unit": "iter/sec",
            "range": "stddev: 0.00007584559417441646",
            "extra": "mean: 145.49461001866516 usec\nrounds: 3613"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 45604.74974734449,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030847976860752266",
            "extra": "mean: 21.927540564088474 usec\nrounds: 24689"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 32175.20885751483,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029126908465184605",
            "extra": "mean: 31.079829331595477 usec\nrounds: 15117"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 37889.17294764725,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024527077871284638",
            "extra": "mean: 26.392764006269914 usec\nrounds: 11619"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 43637.83578249099,
            "unit": "iter/sec",
            "range": "stddev: 0.000003405572600944022",
            "extra": "mean: 22.915893560450918 usec\nrounds: 14487"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 37532.139607078614,
            "unit": "iter/sec",
            "range": "stddev: 0.000002891396568630986",
            "extra": "mean: 26.643831406067736 usec\nrounds: 13577"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 23984.518229367983,
            "unit": "iter/sec",
            "range": "stddev: 0.000003377684256279151",
            "extra": "mean: 41.69356209021302 usec\nrounds: 10630"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 16150.191453818152,
            "unit": "iter/sec",
            "range": "stddev: 0.000002219519793607646",
            "extra": "mean: 61.91877061392884 usec\nrounds: 11400"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 8162.434306352151,
            "unit": "iter/sec",
            "range": "stddev: 0.000008121210920126105",
            "extra": "mean: 122.51247145986612 usec\nrounds: 5413"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2511.5629577531877,
            "unit": "iter/sec",
            "range": "stddev: 0.000010276272748381712",
            "extra": "mean: 398.1584442918315 usec\nrounds: 700"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 883269.5701753241,
            "unit": "iter/sec",
            "range": "stddev: 6.825644647860312e-8",
            "extra": "mean: 1.1321571961338015 usec\nrounds: 170620"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3764.6280494791617,
            "unit": "iter/sec",
            "range": "stddev: 0.00005657921136205595",
            "extra": "mean: 265.63049173964225 usec\nrounds: 787"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 3883.354880776887,
            "unit": "iter/sec",
            "range": "stddev: 0.000007581345574625494",
            "extra": "mean: 257.5093007724147 usec\nrounds: 3235"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1250.9616724529662,
            "unit": "iter/sec",
            "range": "stddev: 0.003419420120907063",
            "extra": "mean: 799.3850027708168 usec\nrounds: 1445"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 617.8579925406403,
            "unit": "iter/sec",
            "range": "stddev: 0.0027579456558482107",
            "extra": "mean: 1.618494884055779 msec\nrounds: 621"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 614.0658317727728,
            "unit": "iter/sec",
            "range": "stddev: 0.0029329002117879635",
            "extra": "mean: 1.628489891894909 msec\nrounds: 629"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 9569.727887709212,
            "unit": "iter/sec",
            "range": "stddev: 0.000009889334585832996",
            "extra": "mean: 104.49617917394917 usec\nrounds: 2545"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 11880.492269062843,
            "unit": "iter/sec",
            "range": "stddev: 0.000042391038753626916",
            "extra": "mean: 84.17159637433795 usec\nrounds: 8223"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 183315.8744955102,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018359789745717852",
            "extra": "mean: 5.455064940513333 usec\nrounds: 22867"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1565654.4204529615,
            "unit": "iter/sec",
            "range": "stddev: 7.909730061128203e-8",
            "extra": "mean: 638.7105525564758 nsec\nrounds: 53692"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 272929.26502787584,
            "unit": "iter/sec",
            "range": "stddev: 5.136963780376435e-7",
            "extra": "mean: 3.6639530022471725 usec\nrounds: 24725"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 4143.96172513771,
            "unit": "iter/sec",
            "range": "stddev: 0.00005387648944873443",
            "extra": "mean: 241.31497014895052 usec\nrounds: 1742"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 3118.353174747573,
            "unit": "iter/sec",
            "range": "stddev: 0.00008912330150802514",
            "extra": "mean: 320.68208569125557 usec\nrounds: 2649"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1663.302481005323,
            "unit": "iter/sec",
            "range": "stddev: 0.00007266029786348655",
            "extra": "mean: 601.2135564155392 usec\nrounds: 1028"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 8.536083500095915,
            "unit": "iter/sec",
            "range": "stddev: 0.028041379221576828",
            "extra": "mean: 117.14974437501269 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 2.601421392260535,
            "unit": "iter/sec",
            "range": "stddev: 0.07813700850777099",
            "extra": "mean: 384.40523437498086 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 8.645120000394222,
            "unit": "iter/sec",
            "range": "stddev: 0.025513001179324913",
            "extra": "mean: 115.67219424998143 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 7.764138997903019,
            "unit": "iter/sec",
            "range": "stddev: 0.006936228719238686",
            "extra": "mean: 128.7972819999855 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 2.394987587001303,
            "unit": "iter/sec",
            "range": "stddev: 0.08752428019987409",
            "extra": "mean: 417.538698499925 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_filesystem",
            "value": 9.981728535748955,
            "unit": "iter/sec",
            "range": "stddev: 0.005378136439262517",
            "extra": "mean: 100.18304910001916 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_filesystem",
            "value": 9.561938851912323,
            "unit": "iter/sec",
            "range": "stddev: 0.022437510994176586",
            "extra": "mean: 104.58130045456281 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 38777.4094320674,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015275299513073799",
            "extra": "mean: 25.7882105753315 usec\nrounds: 8947"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 29625.8361240402,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025069627785106124",
            "extra": "mean: 33.75432159325756 usec\nrounds: 7326"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 39206.28083609192,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015094210322604876",
            "extra": "mean: 25.506117353509218 usec\nrounds: 13387"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 26638.715684989937,
            "unit": "iter/sec",
            "range": "stddev: 0.00000237086192312912",
            "extra": "mean: 37.53934731033103 usec\nrounds: 9349"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 33863.17090674581,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021251714051900847",
            "extra": "mean: 29.53060724153249 usec\nrounds: 10495"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 25934.688929848766,
            "unit": "iter/sec",
            "range": "stddev: 0.000006283796566533343",
            "extra": "mean: 38.558395772739715 usec\nrounds: 8232"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 33689.08424057615,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017083601612931046",
            "extra": "mean: 29.683205184769307 usec\nrounds: 7944"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 23835.75264745524,
            "unit": "iter/sec",
            "range": "stddev: 0.000002708284974753383",
            "extra": "mean: 41.95378324278602 usec\nrounds: 9273"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 28710.898826423345,
            "unit": "iter/sec",
            "range": "stddev: 0.000002614867884258289",
            "extra": "mean: 34.829978888702556 usec\nrounds: 8148"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 18871.036012495842,
            "unit": "iter/sec",
            "range": "stddev: 0.000003936372278389724",
            "extra": "mean: 52.99126128198947 usec\nrounds: 7865"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 9623.582450111537,
            "unit": "iter/sec",
            "range": "stddev: 0.000011444551730508991",
            "extra": "mean: 103.91140775111352 usec\nrounds: 3664"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 55758.33934701149,
            "unit": "iter/sec",
            "range": "stddev: 0.000002749384326739743",
            "extra": "mean: 17.934536998609474 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 28073.440231471825,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027514197763148783",
            "extra": "mean: 35.6208570005947 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 47725.50731344348,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016125006111192085",
            "extra": "mean: 20.953156001723983 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 24386.510385178713,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027938827690585445",
            "extra": "mean: 41.0062770033619 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 769.252243028902,
            "unit": "iter/sec",
            "range": "stddev: 0.00003087854422782443",
            "extra": "mean: 1.2999637102942168 msec\nrounds: 573"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 973.2471849284427,
            "unit": "iter/sec",
            "range": "stddev: 0.00009886946557104881",
            "extra": "mean: 1.0274882018523632 msec\nrounds: 862"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6174.287363114735,
            "unit": "iter/sec",
            "range": "stddev: 0.000008570345885385044",
            "extra": "mean: 161.9620113527614 usec\nrounds: 2202"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 6656.3738636134585,
            "unit": "iter/sec",
            "range": "stddev: 0.00000753288323625244",
            "extra": "mean: 150.23194617514213 usec\nrounds: 3753"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 6842.92566316456,
            "unit": "iter/sec",
            "range": "stddev: 0.00000960417315084338",
            "extra": "mean: 146.13632373401273 usec\nrounds: 1980"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4439.82581961863,
            "unit": "iter/sec",
            "range": "stddev: 0.000029368482400684476",
            "extra": "mean: 225.23406111591507 usec\nrounds: 2569"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 2035.434133384396,
            "unit": "iter/sec",
            "range": "stddev: 0.000033507480049125223",
            "extra": "mean: 491.29568164274656 usec\nrounds: 1781"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 2063.749868761957,
            "unit": "iter/sec",
            "range": "stddev: 0.000016625704780366182",
            "extra": "mean: 484.5548460773009 usec\nrounds: 1923"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 2526.3863927093607,
            "unit": "iter/sec",
            "range": "stddev: 0.000020537226152619706",
            "extra": "mean: 395.82227124314693 usec\nrounds: 2389"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 2528.862781645773,
            "unit": "iter/sec",
            "range": "stddev: 0.000017609339741504827",
            "extra": "mean: 395.4346622750343 usec\nrounds: 2428"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1793.764304481018,
            "unit": "iter/sec",
            "range": "stddev: 0.0000340468752723831",
            "extra": "mean: 557.4868434508878 usec\nrounds: 1565"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1790.0724477412266,
            "unit": "iter/sec",
            "range": "stddev: 0.00002433460927577298",
            "extra": "mean: 558.6366078433492 usec\nrounds: 1683"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 2167.5119559287646,
            "unit": "iter/sec",
            "range": "stddev: 0.000022380454976400043",
            "extra": "mean: 461.3584701411746 usec\nrounds: 1976"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 2186.804542154006,
            "unit": "iter/sec",
            "range": "stddev: 0.00002403199554661923",
            "extra": "mean: 457.28823986024764 usec\nrounds: 2022"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 584.788395809907,
            "unit": "iter/sec",
            "range": "stddev: 0.00003936827830351011",
            "extra": "mean: 1.7100202520520993 msec\nrounds: 488"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 595.7480306481006,
            "unit": "iter/sec",
            "range": "stddev: 0.00009144673913934185",
            "extra": "mean: 1.6785619902295323 msec\nrounds: 512"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0ba98ade0102584e4e7010550f1e9652771f0767",
          "message": "[BugFix] Fix colab in tutos (#757)",
          "timestamp": "2024-04-25T14:32:09+01:00",
          "tree_id": "bca42c7cdd0546314947f4f8edbce626a3c8654a",
          "url": "https://github.com/pytorch/tensordict/commit/0ba98ade0102584e4e7010550f1e9652771f0767"
        },
        "date": 1714052191566,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 63153.295399687,
            "unit": "iter/sec",
            "range": "stddev: 7.326503240149622e-7",
            "extra": "mean: 15.834486445579152 usec\nrounds: 8595"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 62707.66431292734,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010792217999297332",
            "extra": "mean: 15.94701398874854 usec\nrounds: 17657"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 54448.35608116674,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013887007454495356",
            "extra": "mean: 18.36602740603021 usec\nrounds: 31526"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 54502.78371865655,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011658134282903177",
            "extra": "mean: 18.347686700958274 usec\nrounds: 36138"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 375507.4161738355,
            "unit": "iter/sec",
            "range": "stddev: 2.2830113354194302e-7",
            "extra": "mean: 2.663063249693756 usec\nrounds: 115795"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 3807.1654689109105,
            "unit": "iter/sec",
            "range": "stddev: 0.000006334877204824347",
            "extra": "mean: 262.66260507087 usec\nrounds: 2603"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 3792.754995009514,
            "unit": "iter/sec",
            "range": "stddev: 0.000006372483859759345",
            "extra": "mean: 263.6605848033407 usec\nrounds: 3685"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 12936.416487876788,
            "unit": "iter/sec",
            "range": "stddev: 0.000007005930433972582",
            "extra": "mean: 77.3011599415602 usec\nrounds: 10235"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 3772.8156773927603,
            "unit": "iter/sec",
            "range": "stddev: 0.000004444621020262167",
            "extra": "mean: 265.05403006887934 usec\nrounds: 3492"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 12623.961362367692,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030625910082001464",
            "extra": "mean: 79.21443763136206 usec\nrounds: 10959"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 3770.9183286557563,
            "unit": "iter/sec",
            "range": "stddev: 0.00000780097047265261",
            "extra": "mean: 265.1873927899352 usec\nrounds: 3717"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 255013.43151436647,
            "unit": "iter/sec",
            "range": "stddev: 2.467368055516448e-7",
            "extra": "mean: 3.9213620790937194 usec\nrounds: 113547"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 7098.096678508942,
            "unit": "iter/sec",
            "range": "stddev: 0.000004126398613849899",
            "extra": "mean: 140.88283737071674 usec\nrounds: 5903"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 6867.824969769572,
            "unit": "iter/sec",
            "range": "stddev: 0.000024840707129731607",
            "extra": "mean: 145.60650633959762 usec\nrounds: 6073"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 8390.575340870986,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034605995617750728",
            "extra": "mean: 119.18133851071465 usec\nrounds: 7533"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 7129.9383548646865,
            "unit": "iter/sec",
            "range": "stddev: 0.000004174864610655489",
            "extra": "mean: 140.25366703454173 usec\nrounds: 6346"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 8453.003546874636,
            "unit": "iter/sec",
            "range": "stddev: 0.000008165437992515575",
            "extra": "mean: 118.30114520296566 usec\nrounds: 7369"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 7018.624696266535,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033936256046141668",
            "extra": "mean: 142.47805564129064 usec\nrounds: 6488"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 916680.7424239173,
            "unit": "iter/sec",
            "range": "stddev: 7.055959274072784e-8",
            "extra": "mean: 1.0908923398518957 usec\nrounds: 156446"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 19837.354331184208,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016350896022223961",
            "extra": "mean: 50.40994798525152 usec\nrounds: 15188"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 19975.008239501272,
            "unit": "iter/sec",
            "range": "stddev: 0.000002156737285167101",
            "extra": "mean: 50.062557572440205 usec\nrounds: 18125"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 22064.584952731977,
            "unit": "iter/sec",
            "range": "stddev: 0.000001931898003332017",
            "extra": "mean: 45.32149605996476 usec\nrounds: 18274"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 19194.032834163485,
            "unit": "iter/sec",
            "range": "stddev: 0.000003430668551038057",
            "extra": "mean: 52.09952533894277 usec\nrounds: 14602"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 21962.01807995288,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016774082944016847",
            "extra": "mean: 45.5331562135817 usec\nrounds: 18033"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 19634.426377462067,
            "unit": "iter/sec",
            "range": "stddev: 0.000002514081017514538",
            "extra": "mean: 50.93095060560967 usec\nrounds: 17917"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 738391.8753061319,
            "unit": "iter/sec",
            "range": "stddev: 1.8408608252418653e-7",
            "extra": "mean: 1.3542944247394477 usec\nrounds: 150558"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 293781.56541775435,
            "unit": "iter/sec",
            "range": "stddev: 3.0218465635900303e-7",
            "extra": "mean: 3.4038895482703633 usec\nrounds: 111895"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 295601.0437264759,
            "unit": "iter/sec",
            "range": "stddev: 3.2636637839447893e-7",
            "extra": "mean: 3.3829379876118266 usec\nrounds: 114719"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 291576.1782252004,
            "unit": "iter/sec",
            "range": "stddev: 4.617288821944316e-7",
            "extra": "mean: 3.4296354595458234 usec\nrounds: 82556"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 294769.01207143546,
            "unit": "iter/sec",
            "range": "stddev: 3.0589489503600707e-7",
            "extra": "mean: 3.39248685936382 usec\nrounds: 70240"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 238443.73848666588,
            "unit": "iter/sec",
            "range": "stddev: 4.518483649337503e-7",
            "extra": "mean: 4.193861438118332 usec\nrounds: 99811"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 238384.26613671787,
            "unit": "iter/sec",
            "range": "stddev: 7.391050081890429e-7",
            "extra": "mean: 4.194907726949064 usec\nrounds: 100311"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 237246.21590378336,
            "unit": "iter/sec",
            "range": "stddev: 4.3807695349602956e-7",
            "extra": "mean: 4.215030348073311 usec\nrounds: 72591"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 241197.32670654776,
            "unit": "iter/sec",
            "range": "stddev: 4.081003753701693e-7",
            "extra": "mean: 4.145982932956168 usec\nrounds: 71190"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 95953.4941805479,
            "unit": "iter/sec",
            "range": "stddev: 9.174993521104988e-7",
            "extra": "mean: 10.421715316780242 usec\nrounds: 65536"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 101126.38330678156,
            "unit": "iter/sec",
            "range": "stddev: 5.498956466417146e-7",
            "extra": "mean: 9.888616276984363 usec\nrounds: 77496"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 96990.52283057422,
            "unit": "iter/sec",
            "range": "stddev: 8.535983729295828e-7",
            "extra": "mean: 10.310285694065472 usec\nrounds: 56361"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 102177.33092111199,
            "unit": "iter/sec",
            "range": "stddev: 5.514334215905506e-7",
            "extra": "mean: 9.786906655176475 usec\nrounds: 60314"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 88486.04644767927,
            "unit": "iter/sec",
            "range": "stddev: 8.817049004939988e-7",
            "extra": "mean: 11.301216860121421 usec\nrounds: 51162"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 97213.62085451458,
            "unit": "iter/sec",
            "range": "stddev: 5.405492758777927e-7",
            "extra": "mean: 10.286624355825136 usec\nrounds: 67532"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 91400.31585971889,
            "unit": "iter/sec",
            "range": "stddev: 9.17868498699705e-7",
            "extra": "mean: 10.940881227749792 usec\nrounds: 51081"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 98088.62391174084,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010239801705178183",
            "extra": "mean: 10.194862157509622 usec\nrounds: 57232"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2887.798705084124,
            "unit": "iter/sec",
            "range": "stddev: 0.00013968972205404177",
            "extra": "mean: 346.28452400073684 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3297.169549082335,
            "unit": "iter/sec",
            "range": "stddev: 0.000013778157886153366",
            "extra": "mean: 303.2904390004205 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2533.3487431850804,
            "unit": "iter/sec",
            "range": "stddev: 0.0016246764610943632",
            "extra": "mean: 394.73444100030974 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3219.7749303326295,
            "unit": "iter/sec",
            "range": "stddev: 0.000014984565025667014",
            "extra": "mean: 310.5807149994462 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 10543.596143253675,
            "unit": "iter/sec",
            "range": "stddev: 0.0000061261511899442314",
            "extra": "mean: 94.8443004088174 usec\nrounds: 7816"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 2424.226991202236,
            "unit": "iter/sec",
            "range": "stddev: 0.000010255190354122885",
            "extra": "mean: 412.50262604496226 usec\nrounds: 2273"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1467.2590794263137,
            "unit": "iter/sec",
            "range": "stddev: 0.00011743315654319198",
            "extra": "mean: 681.5428945179823 usec\nrounds: 967"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 535380.4126069522,
            "unit": "iter/sec",
            "range": "stddev: 2.177642397231468e-7",
            "extra": "mean: 1.867830754454864 usec\nrounds: 102691"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 109951.89669108043,
            "unit": "iter/sec",
            "range": "stddev: 6.897466071299889e-7",
            "extra": "mean: 9.09488631023427 usec\nrounds: 22491"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 85311.00237970399,
            "unit": "iter/sec",
            "range": "stddev: 6.832839109420524e-7",
            "extra": "mean: 11.72181749253372 usec\nrounds: 24799"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 66277.48459028477,
            "unit": "iter/sec",
            "range": "stddev: 8.049754599197722e-7",
            "extra": "mean: 15.088080155452733 usec\nrounds: 23392"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 73745.47153057794,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020767741527579725",
            "extra": "mean: 13.56015466773927 usec\nrounds: 15071"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 86886.62240837989,
            "unit": "iter/sec",
            "range": "stddev: 8.49605540749637e-7",
            "extra": "mean: 11.509251623337976 usec\nrounds: 12010"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 44410.07986695561,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017251384489743968",
            "extra": "mean: 22.51741052922704 usec\nrounds: 10333"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 17007.901869642214,
            "unit": "iter/sec",
            "range": "stddev: 0.000010980849646209901",
            "extra": "mean: 58.79620000541763 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 53099.84809016611,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014000849135067067",
            "extra": "mean: 18.832445590088163 usec\nrounds: 15714"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 23779.45332387422,
            "unit": "iter/sec",
            "range": "stddev: 0.0000041289587913548576",
            "extra": "mean: 42.05311141429878 usec\nrounds: 8060"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 29668.23613665451,
            "unit": "iter/sec",
            "range": "stddev: 0.000002238939750665102",
            "extra": "mean: 33.70608199941216 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 16735.573718558255,
            "unit": "iter/sec",
            "range": "stddev: 0.0000045811824243914004",
            "extra": "mean: 59.75295599762376 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 11909.900034206566,
            "unit": "iter/sec",
            "range": "stddev: 0.00000443430224244417",
            "extra": "mean: 83.96376099949521 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 20431.468065314788,
            "unit": "iter/sec",
            "range": "stddev: 0.000003346830784369185",
            "extra": "mean: 48.94410899908053 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 51201.781824263824,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018475494546920667",
            "extra": "mean: 19.530570311639305 usec\nrounds: 17266"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 52664.39900277846,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016425014632158207",
            "extra": "mean: 18.988159343605954 usec\nrounds: 18162"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 7070.440130513044,
            "unit": "iter/sec",
            "range": "stddev: 0.00007866782338495242",
            "extra": "mean: 141.43391097881175 usec\nrounds: 3853"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 49195.39706567082,
            "unit": "iter/sec",
            "range": "stddev: 0.00000262142475827848",
            "extra": "mean: 20.32710496604189 usec\nrounds: 26399"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 35431.00423644464,
            "unit": "iter/sec",
            "range": "stddev: 0.000002010619272400032",
            "extra": "mean: 28.22386837603071 usec\nrounds: 19229"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 40487.40321327134,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024349008007347915",
            "extra": "mean: 24.699040210912084 usec\nrounds: 12335"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 48420.61346257808,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016907402701351813",
            "extra": "mean: 20.65236122571745 usec\nrounds: 16970"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 40627.14244792442,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021450153930435274",
            "extra": "mean: 24.614086537880254 usec\nrounds: 15577"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 25686.78915973502,
            "unit": "iter/sec",
            "range": "stddev: 0.000002872515620528517",
            "extra": "mean: 38.93051769847267 usec\nrounds: 11385"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 16730.71375320557,
            "unit": "iter/sec",
            "range": "stddev: 0.000002470229623166561",
            "extra": "mean: 59.77031313493137 usec\nrounds: 11717"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 8330.106623573613,
            "unit": "iter/sec",
            "range": "stddev: 0.0000042017058264582886",
            "extra": "mean: 120.04648261885038 usec\nrounds: 5696"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2546.523721672744,
            "unit": "iter/sec",
            "range": "stddev: 0.000009933103476617104",
            "extra": "mean: 392.69219897277316 usec\nrounds: 2141"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 938781.5127203837,
            "unit": "iter/sec",
            "range": "stddev: 9.919737417056776e-8",
            "extra": "mean: 1.0652105803640772 usec\nrounds: 176648"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3883.0504761732172,
            "unit": "iter/sec",
            "range": "stddev: 0.000009621292359136923",
            "extra": "mean: 257.52948774065624 usec\nrounds: 775"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 4023.045433038497,
            "unit": "iter/sec",
            "range": "stddev: 0.000007106073095786706",
            "extra": "mean: 248.56791121166313 usec\nrounds: 3514"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1315.8811841559436,
            "unit": "iter/sec",
            "range": "stddev: 0.0028301144197971787",
            "extra": "mean: 759.9470317234136 usec\nrounds: 1450"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 617.7643714006584,
            "unit": "iter/sec",
            "range": "stddev: 0.002512643541111537",
            "extra": "mean: 1.6187401642032186 msec\nrounds: 609"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 613.9908528908983,
            "unit": "iter/sec",
            "range": "stddev: 0.0026825655985142317",
            "extra": "mean: 1.6286887586217718 msec\nrounds: 609"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 9869.74359335067,
            "unit": "iter/sec",
            "range": "stddev: 0.000007391947012126037",
            "extra": "mean: 101.31975471720547 usec\nrounds: 2703"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 12403.123923195912,
            "unit": "iter/sec",
            "range": "stddev: 0.000042603473255725236",
            "extra": "mean: 80.6248495292249 usec\nrounds: 8819"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 185091.2910910967,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015590714190214227",
            "extra": "mean: 5.402739340706356 usec\nrounds: 23782"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1555741.95682423,
            "unit": "iter/sec",
            "range": "stddev: 1.0206268114089903e-7",
            "extra": "mean: 642.7801189095148 nsec\nrounds: 70097"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 284263.4490142663,
            "unit": "iter/sec",
            "range": "stddev: 4.1401991598250317e-7",
            "extra": "mean: 3.517863458941614 usec\nrounds: 26688"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 4277.617284846315,
            "unit": "iter/sec",
            "range": "stddev: 0.00005137979357541428",
            "extra": "mean: 233.7750044966745 usec\nrounds: 1779"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 3221.746812169512,
            "unit": "iter/sec",
            "range": "stddev: 0.000051158365229866845",
            "extra": "mean: 310.3906229448874 usec\nrounds: 2737"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1751.1363544747041,
            "unit": "iter/sec",
            "range": "stddev: 0.00005408820812766877",
            "extra": "mean: 571.057757692418 usec\nrounds: 1560"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 9.185822257000261,
            "unit": "iter/sec",
            "range": "stddev: 0.024245120246333452",
            "extra": "mean: 108.8634171250078 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 2.6127086548118337,
            "unit": "iter/sec",
            "range": "stddev: 0.08356613358006115",
            "extra": "mean: 382.7445506249987 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.297779994058114,
            "unit": "iter/sec",
            "range": "stddev: 0.021281371275823645",
            "extra": "mean: 107.55255562500565 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 8.255989639989146,
            "unit": "iter/sec",
            "range": "stddev: 0.005734420951640673",
            "extra": "mean: 121.12418299998191 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 2.447127104882084,
            "unit": "iter/sec",
            "range": "stddev: 0.08165744480321928",
            "extra": "mean: 408.64244362500557 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_filesystem",
            "value": 10.71249738458007,
            "unit": "iter/sec",
            "range": "stddev: 0.006142202605562833",
            "extra": "mean: 93.34891427272912 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_filesystem",
            "value": 10.21951457855696,
            "unit": "iter/sec",
            "range": "stddev: 0.018871778099006697",
            "extra": "mean: 97.85200581818674 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 38636.96313556074,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034399659894692413",
            "extra": "mean: 25.881951345177505 usec\nrounds: 9331"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 30031.215213772324,
            "unit": "iter/sec",
            "range": "stddev: 0.000003428822320978062",
            "extra": "mean: 33.298685813466506 usec\nrounds: 8374"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 39041.34834783484,
            "unit": "iter/sec",
            "range": "stddev: 0.000001425316257018505",
            "extra": "mean: 25.613869456828276 usec\nrounds: 14049"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 27605.27782468404,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022844701084141605",
            "extra": "mean: 36.22495692131096 usec\nrounds: 9796"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 33876.63360420536,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018786756350645225",
            "extra": "mean: 29.518871670763133 usec\nrounds: 10738"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 26325.028696290265,
            "unit": "iter/sec",
            "range": "stddev: 0.000005741926402862749",
            "extra": "mean: 37.986663244964305 usec\nrounds: 11008"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 33713.522408940284,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020909514918829555",
            "extra": "mean: 29.661688502024223 usec\nrounds: 8732"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 23762.32215790058,
            "unit": "iter/sec",
            "range": "stddev: 0.0000042907752054164855",
            "extra": "mean: 42.08342910911661 usec\nrounds: 9564"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 28166.64591285967,
            "unit": "iter/sec",
            "range": "stddev: 0.000002567371778022413",
            "extra": "mean: 35.50298473924591 usec\nrounds: 8191"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 18532.922469303918,
            "unit": "iter/sec",
            "range": "stddev: 0.000004325785140823596",
            "extra": "mean: 53.958030723772794 usec\nrounds: 7356"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 9810.775325510445,
            "unit": "iter/sec",
            "range": "stddev: 0.00001015877813200233",
            "extra": "mean: 101.92874332773195 usec\nrounds: 4009"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 59781.04951467644,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025366482555970564",
            "extra": "mean: 16.72770900006526 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 30002.425095880706,
            "unit": "iter/sec",
            "range": "stddev: 0.000002040083976986069",
            "extra": "mean: 33.330639000155315 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 51362.04437176665,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015297432375666102",
            "extra": "mean: 19.4696300007422 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 26116.916712625814,
            "unit": "iter/sec",
            "range": "stddev: 0.000004679562595550151",
            "extra": "mean: 38.28935900065744 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 759.0883295375129,
            "unit": "iter/sec",
            "range": "stddev: 0.00008300531046474634",
            "extra": "mean: 1.3173697461654648 msec\nrounds: 587"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 977.4861090696722,
            "unit": "iter/sec",
            "range": "stddev: 0.000119763380887728",
            "extra": "mean: 1.023032440790136 msec\nrounds: 912"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6126.759365512128,
            "unit": "iter/sec",
            "range": "stddev: 0.000019856068809344332",
            "extra": "mean: 163.21842271610276 usec\nrounds: 2562"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 6668.276328360487,
            "unit": "iter/sec",
            "range": "stddev: 0.000006634315864972394",
            "extra": "mean: 149.9637913544395 usec\nrounds: 4002"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 7003.966587224423,
            "unit": "iter/sec",
            "range": "stddev: 0.000008404463773662888",
            "extra": "mean: 142.77623794266077 usec\nrounds: 2177"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4543.441034866084,
            "unit": "iter/sec",
            "range": "stddev: 0.00002054890042564823",
            "extra": "mean: 220.09749710099945 usec\nrounds: 2587"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 2097.773405550463,
            "unit": "iter/sec",
            "range": "stddev: 0.000016041767603881412",
            "extra": "mean: 476.69590879268327 usec\nrounds: 1831"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 2096.062150915835,
            "unit": "iter/sec",
            "range": "stddev: 0.000023465245645439987",
            "extra": "mean: 477.08509004042116 usec\nrounds: 1988"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 2552.600043336393,
            "unit": "iter/sec",
            "range": "stddev: 0.000020781492219499253",
            "extra": "mean: 391.75741715217686 usec\nrounds: 2402"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 2508.811989633442,
            "unit": "iter/sec",
            "range": "stddev: 0.000029106596173200558",
            "extra": "mean: 398.59503387741233 usec\nrounds: 2391"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1819.1329454855347,
            "unit": "iter/sec",
            "range": "stddev: 0.000033693155651144355",
            "extra": "mean: 549.7124344219359 usec\nrounds: 1685"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1815.3846170908462,
            "unit": "iter/sec",
            "range": "stddev: 0.000041897351500306545",
            "extra": "mean: 550.847457109392 usec\nrounds: 1737"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 2218.2901481462686,
            "unit": "iter/sec",
            "range": "stddev: 0.00001928818161420459",
            "extra": "mean: 450.7976564002044 usec\nrounds: 2078"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 2208.601110066035,
            "unit": "iter/sec",
            "range": "stddev: 0.000023082722888503025",
            "extra": "mean: 452.7752863304959 usec\nrounds: 2085"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 594.3991216787614,
            "unit": "iter/sec",
            "range": "stddev: 0.0000213949774290961",
            "extra": "mean: 1.6823712612086306 msec\nrounds: 513"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 603.1288841189265,
            "unit": "iter/sec",
            "range": "stddev: 0.000023881628383020772",
            "extra": "mean: 1.6580204104481544 msec\nrounds: 536"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3f27d790cae01bf98ae2d298d344dcfceb6a89af",
          "message": "[Feature] Mean, std, var, prod, sum (#751)",
          "timestamp": "2024-04-25T15:16:37+01:00",
          "tree_id": "f5c73993e4dec4807de4d048335d0f49bb4de8da",
          "url": "https://github.com/pytorch/tensordict/commit/3f27d790cae01bf98ae2d298d344dcfceb6a89af"
        },
        "date": 1714054871534,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 60396.33596262918,
            "unit": "iter/sec",
            "range": "stddev: 6.642731285064978e-7",
            "extra": "mean: 16.55729580381763 usec\nrounds: 8127"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 58678.12741305562,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011373858447542695",
            "extra": "mean: 17.042125304385642 usec\nrounds: 17661"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 51723.9431270531,
            "unit": "iter/sec",
            "range": "stddev: 0.000001133599120976928",
            "extra": "mean: 19.33340614700683 usec\nrounds: 32471"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 51936.23281518527,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012052078426862212",
            "extra": "mean: 19.254380724117848 usec\nrounds: 34634"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 391070.95828179165,
            "unit": "iter/sec",
            "range": "stddev: 1.7636649940065666e-7",
            "extra": "mean: 2.5570807006319196 usec\nrounds: 103221"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 3773.7000135637777,
            "unit": "iter/sec",
            "range": "stddev: 0.000011681331995657669",
            "extra": "mean: 264.9919167940505 usec\nrounds: 3269"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 3581.2656156927474,
            "unit": "iter/sec",
            "range": "stddev: 0.00003875896666348977",
            "extra": "mean: 279.23089413365494 usec\nrounds: 3665"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 12617.914940934797,
            "unit": "iter/sec",
            "range": "stddev: 0.00001782369225216628",
            "extra": "mean: 79.25239666625261 usec\nrounds: 9779"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 3701.0468576920152,
            "unit": "iter/sec",
            "range": "stddev: 0.000017907064176357384",
            "extra": "mean: 270.1938231129565 usec\nrounds: 3392"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 13006.44621301047,
            "unit": "iter/sec",
            "range": "stddev: 0.000001979337485159587",
            "extra": "mean: 76.88495255527145 usec\nrounds: 11192"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 3707.2549678718974,
            "unit": "iter/sec",
            "range": "stddev: 0.000011773944830012399",
            "extra": "mean: 269.7413608360574 usec\nrounds: 3636"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 256776.99768077207,
            "unit": "iter/sec",
            "range": "stddev: 2.4197430754590495e-7",
            "extra": "mean: 3.894429832236028 usec\nrounds: 109566"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 7281.913050979766,
            "unit": "iter/sec",
            "range": "stddev: 0.000007421431461397984",
            "extra": "mean: 137.32655045441007 usec\nrounds: 6164"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 6979.957300581761,
            "unit": "iter/sec",
            "range": "stddev: 0.000023770549633203653",
            "extra": "mean: 143.26735206770573 usec\nrounds: 6530"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 8562.915551320533,
            "unit": "iter/sec",
            "range": "stddev: 0.00001077570427198403",
            "extra": "mean: 116.78265352573572 usec\nrounds: 6722"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 7187.011757630345,
            "unit": "iter/sec",
            "range": "stddev: 0.000005900345437040664",
            "extra": "mean: 139.13988646787934 usec\nrounds: 6518"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 8652.677594213124,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028400876321628943",
            "extra": "mean: 115.57116154065372 usec\nrounds: 7218"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 7056.785825410758,
            "unit": "iter/sec",
            "range": "stddev: 0.000005793957477511934",
            "extra": "mean: 141.7075740628408 usec\nrounds: 6616"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 853174.988252214,
            "unit": "iter/sec",
            "range": "stddev: 9.880380027924551e-8",
            "extra": "mean: 1.1720924942356394 usec\nrounds: 191939"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 19500.484626008853,
            "unit": "iter/sec",
            "range": "stddev: 0.000004749601205803448",
            "extra": "mean: 51.280776820605055 usec\nrounds: 14858"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 19077.148366318925,
            "unit": "iter/sec",
            "range": "stddev: 0.000008921681101157534",
            "extra": "mean: 52.418735798350205 usec\nrounds: 10034"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 21513.91627832636,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026270942293010757",
            "extra": "mean: 46.481541857045535 usec\nrounds: 16628"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 19782.532909477028,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019514744225544144",
            "extra": "mean: 50.54964420258538 usec\nrounds: 16723"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 21801.423302564883,
            "unit": "iter/sec",
            "range": "stddev: 0.000002362136319832559",
            "extra": "mean: 45.8685649153169 usec\nrounds: 18139"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 19610.4117951166,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016791356079883801",
            "extra": "mean: 50.9933197960188 usec\nrounds: 18221"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 738952.9417697483,
            "unit": "iter/sec",
            "range": "stddev: 1.6679687815697523e-7",
            "extra": "mean: 1.3532661465628102 usec\nrounds: 143823"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 293964.524531636,
            "unit": "iter/sec",
            "range": "stddev: 2.9807169384663386e-7",
            "extra": "mean: 3.401771018435871 usec\nrounds: 103445"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 289594.4150423413,
            "unit": "iter/sec",
            "range": "stddev: 4.96313798792895e-7",
            "extra": "mean: 3.453105267426484 usec\nrounds: 123381"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 263844.8270872775,
            "unit": "iter/sec",
            "range": "stddev: 0.000001294830374728454",
            "extra": "mean: 3.7901065222294807 usec\nrounds: 78040"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 280709.5737523068,
            "unit": "iter/sec",
            "range": "stddev: 7.466319488321678e-7",
            "extra": "mean: 3.562400764009504 usec\nrounds: 81150"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 239984.77929988244,
            "unit": "iter/sec",
            "range": "stddev: 4.638567866619113e-7",
            "extra": "mean: 4.166930931692175 usec\nrounds: 100929"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 238222.70850358767,
            "unit": "iter/sec",
            "range": "stddev: 3.304472886918917e-7",
            "extra": "mean: 4.197752625186611 usec\nrounds: 95612"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 241271.6405988579,
            "unit": "iter/sec",
            "range": "stddev: 3.24528314359437e-7",
            "extra": "mean: 4.14470593194422 usec\nrounds: 73449"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 241199.86539226025,
            "unit": "iter/sec",
            "range": "stddev: 3.19959165411756e-7",
            "extra": "mean: 4.145939295503805 usec\nrounds: 77556"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 94687.49102782806,
            "unit": "iter/sec",
            "range": "stddev: 5.136208748244456e-7",
            "extra": "mean: 10.561057106329983 usec\nrounds: 67173"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 98738.73480350092,
            "unit": "iter/sec",
            "range": "stddev: 0.0000134037767366268",
            "extra": "mean: 10.127737629918908 usec\nrounds: 73282"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 94284.81373036854,
            "unit": "iter/sec",
            "range": "stddev: 6.56251883416314e-7",
            "extra": "mean: 10.60616190916763 usec\nrounds: 55982"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 98779.09266128743,
            "unit": "iter/sec",
            "range": "stddev: 8.606635428922766e-7",
            "extra": "mean: 10.123599772565138 usec\nrounds: 61535"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 89790.11671913473,
            "unit": "iter/sec",
            "range": "stddev: 0.00000101023806962644",
            "extra": "mean: 11.137083195114 usec\nrounds: 51878"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 95483.65208976743,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015688102859935786",
            "extra": "mean: 10.472996980256536 usec\nrounds: 42386"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 92080.20894279718,
            "unit": "iter/sec",
            "range": "stddev: 5.988672787830559e-7",
            "extra": "mean: 10.860096990236286 usec\nrounds: 51531"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 97620.77128395272,
            "unit": "iter/sec",
            "range": "stddev: 8.013079936166383e-7",
            "extra": "mean: 10.243721565068027 usec\nrounds: 57931"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2880.7368076741022,
            "unit": "iter/sec",
            "range": "stddev: 0.00014314016134966753",
            "extra": "mean: 347.13341299908507 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3218.899990170231,
            "unit": "iter/sec",
            "range": "stddev: 0.000008764466816507417",
            "extra": "mean: 310.66513500070414 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2525.7867482780857,
            "unit": "iter/sec",
            "range": "stddev: 0.0015378641175198503",
            "extra": "mean: 395.916243001011 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3131.6425563858006,
            "unit": "iter/sec",
            "range": "stddev: 0.000007841271629812576",
            "extra": "mean: 319.32124499996917 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 10401.657695398679,
            "unit": "iter/sec",
            "range": "stddev: 0.00000715788909298973",
            "extra": "mean: 96.13852227057656 usec\nrounds: 5725"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 2449.599068740051,
            "unit": "iter/sec",
            "range": "stddev: 0.00002113258211922212",
            "extra": "mean: 408.23007028425644 usec\nrounds: 2248"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1392.8122753643536,
            "unit": "iter/sec",
            "range": "stddev: 0.00013933113690610716",
            "extra": "mean: 717.9718456591032 usec\nrounds: 933"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 524053.2580019395,
            "unit": "iter/sec",
            "range": "stddev: 1.8669184394523647e-7",
            "extra": "mean: 1.9082030017573117 usec\nrounds: 118260"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 97820.34799696159,
            "unit": "iter/sec",
            "range": "stddev: 6.493120988794863e-7",
            "extra": "mean: 10.222821943253168 usec\nrounds: 24363"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 76219.86499812921,
            "unit": "iter/sec",
            "range": "stddev: 8.731925224804521e-7",
            "extra": "mean: 13.11993927074713 usec\nrounds: 26330"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 60324.82558091046,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013726692491384674",
            "extra": "mean: 16.576923188261084 usec\nrounds: 24163"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 74058.49272486566,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014509236997261963",
            "extra": "mean: 13.502840298344918 usec\nrounds: 14621"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 85954.34759447095,
            "unit": "iter/sec",
            "range": "stddev: 7.93401905859138e-7",
            "extra": "mean: 11.634082835669448 usec\nrounds: 12060"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 44420.38592623479,
            "unit": "iter/sec",
            "range": "stddev: 0.000001698910222323967",
            "extra": "mean: 22.512186221448328 usec\nrounds: 12120"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 16588.534665893254,
            "unit": "iter/sec",
            "range": "stddev: 0.000012775982690418846",
            "extra": "mean: 60.28260000903174 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 52171.65517227315,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012117421783589413",
            "extra": "mean: 19.167496156638215 usec\nrounds: 15872"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 24770.81178278528,
            "unit": "iter/sec",
            "range": "stddev: 0.000003149873429574111",
            "extra": "mean: 40.3700939948589 usec\nrounds: 7777"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 29229.639122908957,
            "unit": "iter/sec",
            "range": "stddev: 0.000002163672196332359",
            "extra": "mean: 34.21184900008711 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 16511.064196553554,
            "unit": "iter/sec",
            "range": "stddev: 0.000004038250585518211",
            "extra": "mean: 60.56544799872654 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 11878.652485185994,
            "unit": "iter/sec",
            "range": "stddev: 0.000004376242476266349",
            "extra": "mean: 84.18463300000667 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 20452.00777901024,
            "unit": "iter/sec",
            "range": "stddev: 0.000003444814199530697",
            "extra": "mean: 48.89495499929808 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 49721.25590338915,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017031598962937217",
            "extra": "mean: 20.112122709511787 usec\nrounds: 17521"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 51341.89549145296,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016653806489383199",
            "extra": "mean: 19.477270763532154 usec\nrounds: 19397"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 7008.459949951035,
            "unit": "iter/sec",
            "range": "stddev: 0.00008846058281838154",
            "extra": "mean: 142.68469922653784 usec\nrounds: 3880"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 47285.45936827936,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019417532964691226",
            "extra": "mean: 21.14815026352124 usec\nrounds: 24670"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 34500.7287939566,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021693608774831567",
            "extra": "mean: 28.98489495604995 usec\nrounds: 16079"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 40083.15182494797,
            "unit": "iter/sec",
            "range": "stddev: 0.000001947854396313259",
            "extra": "mean: 24.94813792007231 usec\nrounds: 12036"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 46758.97708170488,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018282467397750338",
            "extra": "mean: 21.386267673320518 usec\nrounds: 17215"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 39877.63145062629,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018165770492454045",
            "extra": "mean: 25.07671503103514 usec\nrounds: 15342"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 24991.45757905378,
            "unit": "iter/sec",
            "range": "stddev: 0.000002652798949027939",
            "extra": "mean: 40.01367254537948 usec\nrounds: 11284"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 16363.282504081435,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022107143061857665",
            "extra": "mean: 61.11243265222449 usec\nrounds: 11448"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 8244.741010038293,
            "unit": "iter/sec",
            "range": "stddev: 0.0000043574699802065915",
            "extra": "mean: 121.28943756783399 usec\nrounds: 5542"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2530.0061079780126,
            "unit": "iter/sec",
            "range": "stddev: 0.000008827923275662016",
            "extra": "mean: 395.2559627609764 usec\nrounds: 2202"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 896589.32111905,
            "unit": "iter/sec",
            "range": "stddev: 7.374365339530219e-7",
            "extra": "mean: 1.1153378435869847 usec\nrounds: 174521"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3852.383228410484,
            "unit": "iter/sec",
            "range": "stddev: 0.000010456556579293055",
            "extra": "mean: 259.57957469683146 usec\nrounds: 743"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 3920.5245053128465,
            "unit": "iter/sec",
            "range": "stddev: 0.000006861261685821579",
            "extra": "mean: 255.06791212371286 usec\nrounds: 3357"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1299.2350770261116,
            "unit": "iter/sec",
            "range": "stddev: 0.002923942750138158",
            "extra": "mean: 769.6836528528411 usec\nrounds: 1472"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 612.4177401591363,
            "unit": "iter/sec",
            "range": "stddev: 0.0027088745914202964",
            "extra": "mean: 1.6328723588904377 msec\nrounds: 613"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 613.9772954272026,
            "unit": "iter/sec",
            "range": "stddev: 0.002649582863869571",
            "extra": "mean: 1.6287247223111148 msec\nrounds: 623"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 9512.595252336114,
            "unit": "iter/sec",
            "range": "stddev: 0.00006230849856116311",
            "extra": "mean: 105.12378309740645 usec\nrounds: 2674"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 12278.997625301847,
            "unit": "iter/sec",
            "range": "stddev: 0.0000070072445433140645",
            "extra": "mean: 81.43987241592268 usec\nrounds: 8755"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 174086.30434319237,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016440305439046625",
            "extra": "mean: 5.744277263928861 usec\nrounds: 22603"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1569543.9379892282,
            "unit": "iter/sec",
            "range": "stddev: 1.407563768761509e-7",
            "extra": "mean: 637.1277514416822 nsec\nrounds: 68555"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 274442.6391463151,
            "unit": "iter/sec",
            "range": "stddev: 4.2971444995581653e-7",
            "extra": "mean: 3.6437486649691646 usec\nrounds: 29025"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 4117.88535498026,
            "unit": "iter/sec",
            "range": "stddev: 0.00005031493368815504",
            "extra": "mean: 242.84308906040292 usec\nrounds: 1819"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 3143.68900159232,
            "unit": "iter/sec",
            "range": "stddev: 0.000059473201308176886",
            "extra": "mean: 318.0976233633438 usec\nrounds: 2825"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1658.4410693350276,
            "unit": "iter/sec",
            "range": "stddev: 0.00006083032578922817",
            "extra": "mean: 602.9759021832247 usec\nrounds: 1237"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 8.996209874289553,
            "unit": "iter/sec",
            "range": "stddev: 0.024822192832728816",
            "extra": "mean: 111.15792249999856 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 2.58244500049867,
            "unit": "iter/sec",
            "range": "stddev: 0.07176998600404588",
            "extra": "mean: 387.2299312499976 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.312671602199247,
            "unit": "iter/sec",
            "range": "stddev: 0.02172649180110916",
            "extra": "mean: 107.38057162499359 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 8.096937713518972,
            "unit": "iter/sec",
            "range": "stddev: 0.007750624572716096",
            "extra": "mean: 123.5034818571421 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 1.5533955115886817,
            "unit": "iter/sec",
            "range": "stddev: 0.2984458925952905",
            "extra": "mean: 643.7510553749987 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_filesystem",
            "value": 10.657183632658645,
            "unit": "iter/sec",
            "range": "stddev: 0.004547835955838694",
            "extra": "mean: 93.83342114285499 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_filesystem",
            "value": 10.768932828254162,
            "unit": "iter/sec",
            "range": "stddev: 0.00265547876861946",
            "extra": "mean: 92.85971190908785 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 39015.5894593081,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012468571727638451",
            "extra": "mean: 25.63078025626051 usec\nrounds: 9593"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 30429.93566685747,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037587749492539735",
            "extra": "mean: 32.86237640946255 usec\nrounds: 7537"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 38954.569879067116,
            "unit": "iter/sec",
            "range": "stddev: 0.000003768843559232824",
            "extra": "mean: 25.670929061839463 usec\nrounds: 15295"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 27708.339443368648,
            "unit": "iter/sec",
            "range": "stddev: 0.000004240030399347422",
            "extra": "mean: 36.090217605563765 usec\nrounds: 9338"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 31364.42403226488,
            "unit": "iter/sec",
            "range": "stddev: 0.000007314805455498161",
            "extra": "mean: 31.88325725258945 usec\nrounds: 10410"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 25665.64468400458,
            "unit": "iter/sec",
            "range": "stddev: 0.000008464335025201828",
            "extra": "mean: 38.96259035422644 usec\nrounds: 8439"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 34013.17367691464,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016869718131748882",
            "extra": "mean: 29.400373205359493 usec\nrounds: 8360"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 24030.517709393516,
            "unit": "iter/sec",
            "range": "stddev: 0.000004621190069728959",
            "extra": "mean: 41.61375181730274 usec\nrounds: 9767"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 28289.133257725207,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025721296164377942",
            "extra": "mean: 35.34926259103112 usec\nrounds: 8359"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 18138.622460506896,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037048217451281197",
            "extra": "mean: 55.13097823041929 usec\nrounds: 7809"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 9908.256155239793,
            "unit": "iter/sec",
            "range": "stddev: 0.00000872863425166382",
            "extra": "mean: 100.92593331584075 usec\nrounds: 3839"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 55947.586508334876,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014442583271021109",
            "extra": "mean: 17.873872000734536 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 28777.37674631717,
            "unit": "iter/sec",
            "range": "stddev: 0.000002230752140677672",
            "extra": "mean: 34.74951899943335 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 47533.509934928436,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014084852573010719",
            "extra": "mean: 21.037790000548284 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 24914.651745702075,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019088756421156073",
            "extra": "mean: 40.13702500066074 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 759.5489028948186,
            "unit": "iter/sec",
            "range": "stddev: 0.0000616938789860421",
            "extra": "mean: 1.3165709228053202 msec\nrounds: 570"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 991.7594226605801,
            "unit": "iter/sec",
            "range": "stddev: 0.00009082785083073101",
            "extra": "mean: 1.0083090486978314 msec\nrounds: 883"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6111.503993895654,
            "unit": "iter/sec",
            "range": "stddev: 0.000011343545814513386",
            "extra": "mean: 163.62584414553743 usec\nrounds: 2528"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 6431.223154308526,
            "unit": "iter/sec",
            "range": "stddev: 0.00001151671440957861",
            "extra": "mean: 155.4914167968283 usec\nrounds: 3834"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 6614.576495531696,
            "unit": "iter/sec",
            "range": "stddev: 0.000009166147314769272",
            "extra": "mean: 151.18125864528497 usec\nrounds: 2169"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4447.147765162844,
            "unit": "iter/sec",
            "range": "stddev: 0.000011311153506672675",
            "extra": "mean: 224.86322758006725 usec\nrounds: 2654"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 2031.6827073337818,
            "unit": "iter/sec",
            "range": "stddev: 0.000028924600367669925",
            "extra": "mean: 492.20284072423897 usec\nrounds: 1601"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 2042.180531980391,
            "unit": "iter/sec",
            "range": "stddev: 0.00002878801823855526",
            "extra": "mean: 489.6726730766827 usec\nrounds: 1924"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 2508.0164477906565,
            "unit": "iter/sec",
            "range": "stddev: 0.000017103872171156865",
            "extra": "mean: 398.7214680672899 usec\nrounds: 2380"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 2510.426115449836,
            "unit": "iter/sec",
            "range": "stddev: 0.000013287691409590886",
            "extra": "mean: 398.33874968306446 usec\nrounds: 2369"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1791.026327239679,
            "unit": "iter/sec",
            "range": "stddev: 0.00002070244226620444",
            "extra": "mean: 558.3390845745943 usec\nrounds: 1679"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1786.1674349752125,
            "unit": "iter/sec",
            "range": "stddev: 0.00002287702569547384",
            "extra": "mean: 559.8579284443607 usec\nrounds: 1691"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 2173.574666292166,
            "unit": "iter/sec",
            "range": "stddev: 0.0000229148188112236",
            "extra": "mean: 460.0716117591991 usec\nrounds: 2058"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 2177.704231157014,
            "unit": "iter/sec",
            "range": "stddev: 0.00002090544975587753",
            "extra": "mean: 459.1991812720592 usec\nrounds: 1997"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 596.152690776475,
            "unit": "iter/sec",
            "range": "stddev: 0.000029802030187097458",
            "extra": "mean: 1.677422605771557 msec\nrounds: 520"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 605.9407330380166,
            "unit": "iter/sec",
            "range": "stddev: 0.000029458663478545512",
            "extra": "mean: 1.6503264188665465 msec\nrounds: 530"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e7e1f50eb4479b42fe2df8a160e57b4e84e93dfe",
          "message": "[CI] Remove all osx x86 workflows (#760)",
          "timestamp": "2024-04-25T15:24:16+01:00",
          "tree_id": "0b8e05530ae015abdde10055b0e1b5c6b4730b91",
          "url": "https://github.com/pytorch/tensordict/commit/e7e1f50eb4479b42fe2df8a160e57b4e84e93dfe"
        },
        "date": 1714055317311,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 60982.30442019181,
            "unit": "iter/sec",
            "range": "stddev: 9.16052604244876e-7",
            "extra": "mean: 16.398199600815524 usec\nrounds: 8011"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 60105.5108772699,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010258947316858983",
            "extra": "mean: 16.63740953873449 usec\nrounds: 17549"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 52374.022208588176,
            "unit": "iter/sec",
            "range": "stddev: 0.000001302434716383742",
            "extra": "mean: 19.09343521521672 usec\nrounds: 32662"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 53035.51030503724,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013075343224130732",
            "extra": "mean: 18.855291374560817 usec\nrounds: 34526"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 368379.1980196319,
            "unit": "iter/sec",
            "range": "stddev: 2.2177398957951723e-7",
            "extra": "mean: 2.7145941067679598 usec\nrounds: 107447"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 3705.404909270088,
            "unit": "iter/sec",
            "range": "stddev: 0.00002082891069011477",
            "extra": "mean: 269.8760390526351 usec\nrounds: 3252"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 3749.4409870221316,
            "unit": "iter/sec",
            "range": "stddev: 0.00000441877877677549",
            "extra": "mean: 266.7064246273727 usec\nrounds: 3622"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 12589.07131837008,
            "unit": "iter/sec",
            "range": "stddev: 0.000019148679011321142",
            "extra": "mean: 79.43397687649853 usec\nrounds: 8260"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 3675.0143445864587,
            "unit": "iter/sec",
            "range": "stddev: 0.000008998892749295133",
            "extra": "mean: 272.1077814221506 usec\nrounds: 3445"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 12901.42233394775,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024872856682724863",
            "extra": "mean: 77.51083362093198 usec\nrounds: 11011"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 3642.313404637703,
            "unit": "iter/sec",
            "range": "stddev: 0.00003548828376567404",
            "extra": "mean: 274.5507837756945 usec\nrounds: 3353"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 253693.97919101827,
            "unit": "iter/sec",
            "range": "stddev: 6.960625498787512e-7",
            "extra": "mean: 3.941756927731629 usec\nrounds: 118977"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 7203.46935815558,
            "unit": "iter/sec",
            "range": "stddev: 0.00001007501630813223",
            "extra": "mean: 138.82199677407195 usec\nrounds: 5890"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 6936.595326303902,
            "unit": "iter/sec",
            "range": "stddev: 0.000021799246152140808",
            "extra": "mean: 144.1629434843852 usec\nrounds: 6423"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 8292.16383044041,
            "unit": "iter/sec",
            "range": "stddev: 0.000015015816566142422",
            "extra": "mean: 120.59578421847083 usec\nrounds: 7452"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 7002.673594946045,
            "unit": "iter/sec",
            "range": "stddev: 0.000020097573456757",
            "extra": "mean: 142.8026005269927 usec\nrounds: 6456"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 8455.078596828685,
            "unit": "iter/sec",
            "range": "stddev: 0.000012605148703675457",
            "extra": "mean: 118.2721116720403 usec\nrounds: 7531"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 6882.753767313926,
            "unit": "iter/sec",
            "range": "stddev: 0.000006525835275437668",
            "extra": "mean: 145.2906836140183 usec\nrounds: 6530"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 842504.9802637253,
            "unit": "iter/sec",
            "range": "stddev: 8.996179757055943e-8",
            "extra": "mean: 1.1869366038488876 usec\nrounds: 195351"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 19685.460524831225,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027990799316302676",
            "extra": "mean: 50.79891317445181 usec\nrounds: 15295"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 19580.25548445332,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036097397936561216",
            "extra": "mean: 51.07185658501738 usec\nrounds: 17927"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 21865.62160162652,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025694508405357383",
            "extra": "mean: 45.7338930591213 usec\nrounds: 16813"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 19469.91658889284,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031962034863391136",
            "extra": "mean: 51.361288346272524 usec\nrounds: 15849"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 21839.276843885018,
            "unit": "iter/sec",
            "range": "stddev: 0.000001612301171882523",
            "extra": "mean: 45.78906193407221 usec\nrounds: 17567"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 19315.854117843388,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018411532425093589",
            "extra": "mean: 51.77094390437702 usec\nrounds: 17381"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 724225.8932138409,
            "unit": "iter/sec",
            "range": "stddev: 1.9744784029946944e-7",
            "extra": "mean: 1.3807846548573093 usec\nrounds: 156962"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 287358.2078473237,
            "unit": "iter/sec",
            "range": "stddev: 3.5838988119287423e-7",
            "extra": "mean: 3.4799771598356783 usec\nrounds: 106304"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 287614.3947551104,
            "unit": "iter/sec",
            "range": "stddev: 3.3173440016635186e-7",
            "extra": "mean: 3.476877438111021 usec\nrounds: 117565"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 289166.3146388011,
            "unit": "iter/sec",
            "range": "stddev: 3.813868639747865e-7",
            "extra": "mean: 3.4582174664746286 usec\nrounds: 63573"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 258689.29666272135,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024704361602868278",
            "extra": "mean: 3.865641187713299 usec\nrounds: 49073"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 233245.05653912429,
            "unit": "iter/sec",
            "range": "stddev: 9.053559615132882e-7",
            "extra": "mean: 4.287336309879138 usec\nrounds: 102691"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 237006.4479845442,
            "unit": "iter/sec",
            "range": "stddev: 7.569957357749712e-7",
            "extra": "mean: 4.219294489680772 usec\nrounds: 101844"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 235789.65327966813,
            "unit": "iter/sec",
            "range": "stddev: 4.6389536714548696e-7",
            "extra": "mean: 4.241068198246632 usec\nrounds: 70339"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 236326.89161888443,
            "unit": "iter/sec",
            "range": "stddev: 3.6942940570022737e-7",
            "extra": "mean: 4.2314270422202425 usec\nrounds: 73234"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 93760.34774078445,
            "unit": "iter/sec",
            "range": "stddev: 6.286327820072143e-7",
            "extra": "mean: 10.665489453651139 usec\nrounds: 69028"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 97991.46350741126,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010751591182400047",
            "extra": "mean: 10.204970557709533 usec\nrounds: 76081"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 94380.45024068368,
            "unit": "iter/sec",
            "range": "stddev: 6.532052927233572e-7",
            "extra": "mean: 10.59541459539403 usec\nrounds: 56168"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 98339.2903208927,
            "unit": "iter/sec",
            "range": "stddev: 7.659921736542569e-7",
            "extra": "mean: 10.16887549967955 usec\nrounds: 60530"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 89365.53287587162,
            "unit": "iter/sec",
            "range": "stddev: 6.334893937843138e-7",
            "extra": "mean: 11.189996498862667 usec\nrounds: 52841"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 96338.18422005027,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011327628736893805",
            "extra": "mean: 10.380100145087395 usec\nrounds: 68980"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 90401.05622893394,
            "unit": "iter/sec",
            "range": "stddev: 8.927727560994745e-7",
            "extra": "mean: 11.061817656948326 usec\nrounds: 51107"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 96799.3424916264,
            "unit": "iter/sec",
            "range": "stddev: 8.551145189609793e-7",
            "extra": "mean: 10.330648682727412 usec\nrounds: 57199"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2837.2491362451015,
            "unit": "iter/sec",
            "range": "stddev: 0.00016276926821477808",
            "extra": "mean: 352.45406800032697 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3213.7311470842774,
            "unit": "iter/sec",
            "range": "stddev: 0.000010977193590874068",
            "extra": "mean: 311.1647970015383 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2485.8901681957705,
            "unit": "iter/sec",
            "range": "stddev: 0.0016449698044202943",
            "extra": "mean: 402.2703870001578 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3128.3564331835555,
            "unit": "iter/sec",
            "range": "stddev: 0.000012375550830341952",
            "extra": "mean: 319.656669998551 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 10489.810246142728,
            "unit": "iter/sec",
            "range": "stddev: 0.000009053502196567688",
            "extra": "mean: 95.33060908968454 usec\nrounds: 7723"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 2449.755003819229,
            "unit": "iter/sec",
            "range": "stddev: 0.0000069916620513423445",
            "extra": "mean: 408.2040850782936 usec\nrounds: 2292"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1445.0668278567166,
            "unit": "iter/sec",
            "range": "stddev: 0.00006893666376844997",
            "extra": "mean: 692.0095186761517 usec\nrounds: 937"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 506560.46572641924,
            "unit": "iter/sec",
            "range": "stddev: 3.440143008637983e-7",
            "extra": "mean: 1.9740979955196016 usec\nrounds: 95239"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 105742.19173850947,
            "unit": "iter/sec",
            "range": "stddev: 7.751722489172795e-7",
            "extra": "mean: 9.456963049081736 usec\nrounds: 19404"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 82288.07444253274,
            "unit": "iter/sec",
            "range": "stddev: 8.102807039646165e-7",
            "extra": "mean: 12.152429216196651 usec\nrounds: 21933"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 64517.822454391804,
            "unit": "iter/sec",
            "range": "stddev: 8.549067238284202e-7",
            "extra": "mean: 15.499593166010964 usec\nrounds: 21306"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 73078.73248100028,
            "unit": "iter/sec",
            "range": "stddev: 0.000002040353047672396",
            "extra": "mean: 13.683871710007969 usec\nrounds: 14249"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 83423.21649292232,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011511471680222097",
            "extra": "mean: 11.987070770459212 usec\nrounds: 11177"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 42232.54985826389,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018298618476712739",
            "extra": "mean: 23.67841873995501 usec\nrounds: 9968"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 16400.056412107144,
            "unit": "iter/sec",
            "range": "stddev: 0.000013534418878468449",
            "extra": "mean: 60.97540001519519 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 51229.82614794813,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015261556715359613",
            "extra": "mean: 19.51987885166876 usec\nrounds: 15188"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 23304.1184074312,
            "unit": "iter/sec",
            "range": "stddev: 0.000003839361385556233",
            "extra": "mean: 42.91087019542094 usec\nrounds: 7388"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 29061.381036584644,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035539175439800592",
            "extra": "mean: 34.40992700041079 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 16287.820114738244,
            "unit": "iter/sec",
            "range": "stddev: 0.0000038723702048464225",
            "extra": "mean: 61.39556999988826 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 11974.19125921813,
            "unit": "iter/sec",
            "range": "stddev: 0.0000041833475066019",
            "extra": "mean: 83.51294700008793 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 19729.078324168808,
            "unit": "iter/sec",
            "range": "stddev: 0.000006344046541871469",
            "extra": "mean: 50.68660499841826 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 50169.13960416802,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017999457815851067",
            "extra": "mean: 19.932572252383626 usec\nrounds: 18553"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 51425.91253210066,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017507496806671374",
            "extra": "mean: 19.445449789068654 usec\nrounds: 14927"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 6882.36798830329,
            "unit": "iter/sec",
            "range": "stddev: 0.00009931448677247346",
            "extra": "mean: 145.29882762728153 usec\nrounds: 3417"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 49235.27977798872,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020559324157807755",
            "extra": "mean: 20.310639129282723 usec\nrounds: 23338"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 35334.95492238166,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030921782602472995",
            "extra": "mean: 28.300587964429123 usec\nrounds: 16717"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 40298.97133925887,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024664591165232634",
            "extra": "mean: 24.81452917449061 usec\nrounds: 11637"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 47471.93040153434,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017968162229972761",
            "extra": "mean: 21.06507975432318 usec\nrounds: 17742"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 39970.642824738905,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034820108304653743",
            "extra": "mean: 25.018361710737192 usec\nrounds: 14542"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 24769.63713945783,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027034006272499747",
            "extra": "mean: 40.37200845413307 usec\nrounds: 11119"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 16155.148009359587,
            "unit": "iter/sec",
            "range": "stddev: 0.000009070952882952959",
            "extra": "mean: 61.899773336687694 usec\nrounds: 11184"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 8192.919644041222,
            "unit": "iter/sec",
            "range": "stddev: 0.000004880095248021418",
            "extra": "mean: 122.05661027412961 usec\nrounds: 5509"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2493.285986791609,
            "unit": "iter/sec",
            "range": "stddev: 0.000009077956088225374",
            "extra": "mean: 401.0771348724469 usec\nrounds: 2165"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 901776.0022119667,
            "unit": "iter/sec",
            "range": "stddev: 7.712718818444556e-8",
            "extra": "mean: 1.108922833993253 usec\nrounds: 199641"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3840.86475734344,
            "unit": "iter/sec",
            "range": "stddev: 0.00004836083169631991",
            "extra": "mean: 260.3580347597703 usec\nrounds: 748"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 3893.623368743579,
            "unit": "iter/sec",
            "range": "stddev: 0.0000073904083539542975",
            "extra": "mean: 256.8301824022304 usec\nrounds: 3421"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1284.44354722983,
            "unit": "iter/sec",
            "range": "stddev: 0.002986192935684986",
            "extra": "mean: 778.5472566363139 usec\nrounds: 1469"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 602.8126719195665,
            "unit": "iter/sec",
            "range": "stddev: 0.0026474299629236056",
            "extra": "mean: 1.65889014379152 msec\nrounds: 612"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 596.8127349594205,
            "unit": "iter/sec",
            "range": "stddev: 0.0027031961580022304",
            "extra": "mean: 1.6755674626614558 msec\nrounds: 616"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 9541.514471383032,
            "unit": "iter/sec",
            "range": "stddev: 0.000008177376547590785",
            "extra": "mean: 104.8051651547777 usec\nrounds: 2646"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 11698.42627510521,
            "unit": "iter/sec",
            "range": "stddev: 0.00007755607130532707",
            "extra": "mean: 85.48158329022819 usec\nrounds: 7792"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 179719.94827197946,
            "unit": "iter/sec",
            "range": "stddev: 0.000001701095668259086",
            "extra": "mean: 5.56421259640387 usec\nrounds: 21943"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1542039.5901495735,
            "unit": "iter/sec",
            "range": "stddev: 7.820911640550922e-8",
            "extra": "mean: 648.49178087769 nsec\nrounds: 68134"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 275251.1600973631,
            "unit": "iter/sec",
            "range": "stddev: 5.010801296003264e-7",
            "extra": "mean: 3.6330455415565743 usec\nrounds: 25581"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 3990.129992289412,
            "unit": "iter/sec",
            "range": "stddev: 0.0000512954833517785",
            "extra": "mean: 250.61840138853003 usec\nrounds: 1729"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 3065.933099883472,
            "unit": "iter/sec",
            "range": "stddev: 0.000054923696541272884",
            "extra": "mean: 326.1649773238716 usec\nrounds: 2690"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1665.713189581209,
            "unit": "iter/sec",
            "range": "stddev: 0.00008926750693909455",
            "extra": "mean: 600.3434482327767 usec\nrounds: 1555"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 8.914164874630051,
            "unit": "iter/sec",
            "range": "stddev: 0.02483171569468344",
            "extra": "mean: 112.18100787500873 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 2.6241932900161977,
            "unit": "iter/sec",
            "range": "stddev: 0.09084705940893707",
            "extra": "mean: 381.0694905000034 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.078936357074657,
            "unit": "iter/sec",
            "range": "stddev: 0.026709151588242618",
            "extra": "mean: 110.145061125003 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 7.736597066466204,
            "unit": "iter/sec",
            "range": "stddev: 0.008912577227406458",
            "extra": "mean: 129.25579442859154 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 1.4855104432921644,
            "unit": "iter/sec",
            "range": "stddev: 0.296853641658143",
            "extra": "mean: 673.169283000001 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_filesystem",
            "value": 10.598780450973358,
            "unit": "iter/sec",
            "range": "stddev: 0.003967008920682587",
            "extra": "mean: 94.3504778333401 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_filesystem",
            "value": 10.624008910002077,
            "unit": "iter/sec",
            "range": "stddev: 0.003950057097252343",
            "extra": "mean: 94.1264270833339 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 38381.73437522363,
            "unit": "iter/sec",
            "range": "stddev: 0.000002112121827851243",
            "extra": "mean: 26.054059731222704 usec\nrounds: 9007"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 29200.790154571754,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036732630000007035",
            "extra": "mean: 34.24564865219708 usec\nrounds: 7272"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 38539.24203753828,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014410973221593024",
            "extra": "mean: 25.947578289836954 usec\nrounds: 13929"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 26493.444368940538,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020531045827869772",
            "extra": "mean: 37.74518654782182 usec\nrounds: 9649"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 33797.57538401429,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016477818193698285",
            "extra": "mean: 29.587921282453408 usec\nrounds: 10544"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 25654.275704291445,
            "unit": "iter/sec",
            "range": "stddev: 0.000005023076032599666",
            "extra": "mean: 38.97985706268527 usec\nrounds: 8920"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 33606.61997211269,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022293969889491918",
            "extra": "mean: 29.75604213782332 usec\nrounds: 7784"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 23571.79376294724,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022936858903653893",
            "extra": "mean: 42.42358515676099 usec\nrounds: 9688"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 27287.1662810716,
            "unit": "iter/sec",
            "range": "stddev: 0.000002328492558542151",
            "extra": "mean: 36.64726449421295 usec\nrounds: 8072"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 17124.362281096783,
            "unit": "iter/sec",
            "range": "stddev: 0.000004074403583936799",
            "extra": "mean: 58.396335208574655 usec\nrounds: 7467"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 9887.396029361462,
            "unit": "iter/sec",
            "range": "stddev: 0.000007851549817761142",
            "extra": "mean: 101.13886376457614 usec\nrounds: 3883"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 57252.13675214049,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018046200151513628",
            "extra": "mean: 17.466596999327066 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 29904.50743283398,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018124773155529827",
            "extra": "mean: 33.439774998669236 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 50115.86537806771,
            "unit": "iter/sec",
            "range": "stddev: 0.000001574614208294858",
            "extra": "mean: 19.953760998760117 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 25612.59762772909,
            "unit": "iter/sec",
            "range": "stddev: 0.000003457143156527259",
            "extra": "mean: 39.043287000197324 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 762.9067332273343,
            "unit": "iter/sec",
            "range": "stddev: 0.000022424133616988355",
            "extra": "mean: 1.3107762147670226 msec\nrounds: 596"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 920.8001056983343,
            "unit": "iter/sec",
            "range": "stddev: 0.002322128890075281",
            "extra": "mean: 1.0860120386732586 msec\nrounds: 905"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6123.112620916615,
            "unit": "iter/sec",
            "range": "stddev: 0.000009051364820217344",
            "extra": "mean: 163.31563077641098 usec\nrounds: 2294"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 6039.390117413597,
            "unit": "iter/sec",
            "range": "stddev: 0.000034103083046938957",
            "extra": "mean: 165.5796331349192 usec\nrounds: 3519"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 6819.888095769189,
            "unit": "iter/sec",
            "range": "stddev: 0.000010054957825680036",
            "extra": "mean: 146.62997192290644 usec\nrounds: 2137"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4449.316030424747,
            "unit": "iter/sec",
            "range": "stddev: 0.00002121576908758479",
            "extra": "mean: 224.75364599006392 usec\nrounds: 2531"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 2063.8709582899255,
            "unit": "iter/sec",
            "range": "stddev: 0.000024641225223514626",
            "extra": "mean: 484.5264167235418 usec\nrounds: 1495"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 2080.4551290019194,
            "unit": "iter/sec",
            "range": "stddev: 0.000021704485275222927",
            "extra": "mean: 480.6640556961887 usec\nrounds: 1975"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 2537.1612568775386,
            "unit": "iter/sec",
            "range": "stddev: 0.000012720497919215345",
            "extra": "mean: 394.14128577333355 usec\nrounds: 2404"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 2501.3950143126935,
            "unit": "iter/sec",
            "range": "stddev: 0.000016274209922574895",
            "extra": "mean: 399.7769221886649 usec\nrounds: 2339"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1814.5792862283263,
            "unit": "iter/sec",
            "range": "stddev: 0.000028148549334443435",
            "extra": "mean: 551.0919294568489 usec\nrounds: 1616"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1823.6516860215104,
            "unit": "iter/sec",
            "range": "stddev: 0.00002220810652481297",
            "extra": "mean: 548.3503278971029 usec\nrounds: 1717"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 2189.8428617712407,
            "unit": "iter/sec",
            "range": "stddev: 0.00002384941833193688",
            "extra": "mean: 456.65377066880325 usec\nrounds: 2032"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 2208.8613984428184,
            "unit": "iter/sec",
            "range": "stddev: 0.00001218144334324191",
            "extra": "mean: 452.72193208001653 usec\nrounds: 1649"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 589.1894893381981,
            "unit": "iter/sec",
            "range": "stddev: 0.00001665013424879047",
            "extra": "mean: 1.6972468417982833 msec\nrounds: 512"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 600.845908637309,
            "unit": "iter/sec",
            "range": "stddev: 0.000042525333411710984",
            "extra": "mean: 1.6643202285723375 msec\nrounds: 525"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fac63c331d108f6d7928fc3ae454b51e71d1ef27",
          "message": "[CI] Doc on release tag (#761)",
          "timestamp": "2024-04-25T15:28:06+01:00",
          "tree_id": "afa6c879027e57d38e9410a8fef683cc049c80b1",
          "url": "https://github.com/pytorch/tensordict/commit/fac63c331d108f6d7928fc3ae454b51e71d1ef27"
        },
        "date": 1714055555104,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 58762.04387309767,
            "unit": "iter/sec",
            "range": "stddev: 9.571133735792463e-7",
            "extra": "mean: 17.01778791356538 usec\nrounds: 6950"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 57696.07287677019,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015622715215362991",
            "extra": "mean: 17.332202178402053 usec\nrounds: 16436"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 50830.03901263899,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014996668409649107",
            "extra": "mean: 19.673406108371232 usec\nrounds: 28682"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 50510.68601726436,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013744173165915217",
            "extra": "mean: 19.797790900289968 usec\nrounds: 32726"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 387331.80635253043,
            "unit": "iter/sec",
            "range": "stddev: 2.488811818453148e-7",
            "extra": "mean: 2.5817657718763454 usec\nrounds: 104189"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 3716.952599604256,
            "unit": "iter/sec",
            "range": "stddev: 0.000006706563877048881",
            "extra": "mean: 269.03759819441063 usec\nrounds: 3101"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 3623.1215804791536,
            "unit": "iter/sec",
            "range": "stddev: 0.00003041338389517697",
            "extra": "mean: 276.0050905793096 usec\nrounds: 3577"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 12567.163697260345,
            "unit": "iter/sec",
            "range": "stddev: 0.00001777419564462418",
            "extra": "mean: 79.57244960674787 usec\nrounds: 9664"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 3671.430216999734,
            "unit": "iter/sec",
            "range": "stddev: 0.000014861062837985585",
            "extra": "mean: 272.37341877552905 usec\nrounds: 3398"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 12685.338746664396,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037826384048004122",
            "extra": "mean: 78.83116249165593 usec\nrounds: 10819"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 3694.684128624939,
            "unit": "iter/sec",
            "range": "stddev: 0.000005972328429437746",
            "extra": "mean: 270.659132198176 usec\nrounds: 3525"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 262604.32661119255,
            "unit": "iter/sec",
            "range": "stddev: 2.3143960459312532e-7",
            "extra": "mean: 3.8080103740277775 usec\nrounds: 89358"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 7267.0220528918735,
            "unit": "iter/sec",
            "range": "stddev: 0.000005939189327280179",
            "extra": "mean: 137.6079489950158 usec\nrounds: 5921"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 6968.3611334390025,
            "unit": "iter/sec",
            "range": "stddev: 0.000027159207687702206",
            "extra": "mean: 143.50576568159053 usec\nrounds: 6393"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 8529.991769425897,
            "unit": "iter/sec",
            "range": "stddev: 0.000005570765496187745",
            "extra": "mean: 117.23340737376867 usec\nrounds: 7649"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 7228.1119724343625,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035703353429369044",
            "extra": "mean: 138.34871454864984 usec\nrounds: 6495"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 8682.785968490822,
            "unit": "iter/sec",
            "range": "stddev: 0.000002930723538422691",
            "extra": "mean: 115.17040770426969 usec\nrounds: 7606"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 7029.969096653259,
            "unit": "iter/sec",
            "range": "stddev: 0.000004017087534359241",
            "extra": "mean: 142.2481359805789 usec\nrounds: 6523"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 867887.4368130236,
            "unit": "iter/sec",
            "range": "stddev: 7.342043608650133e-8",
            "extra": "mean: 1.1522231542746346 usec\nrounds: 163106"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 20058.646526287917,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016171389349701781",
            "extra": "mean: 49.85381235416094 usec\nrounds: 15428"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 19894.57098201588,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024538226974714935",
            "extra": "mean: 50.2649693177084 usec\nrounds: 17991"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 21992.026987537425,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012112732331738179",
            "extra": "mean: 45.47102459298936 usec\nrounds: 18298"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 19559.584301751678,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016680284062364406",
            "extra": "mean: 51.12583092629653 usec\nrounds: 15579"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 21997.91089012663,
            "unit": "iter/sec",
            "range": "stddev: 0.000002574514826326983",
            "extra": "mean: 45.45886220717587 usec\nrounds: 16953"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 19640.7097127585,
            "unit": "iter/sec",
            "range": "stddev: 0.000001859936177777666",
            "extra": "mean: 50.9146570885066 usec\nrounds: 17818"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 738133.251280815,
            "unit": "iter/sec",
            "range": "stddev: 1.4292388586254447e-7",
            "extra": "mean: 1.3547689367262505 usec\nrounds: 115661"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 284852.8572728714,
            "unit": "iter/sec",
            "range": "stddev: 4.097247290163425e-7",
            "extra": "mean: 3.510584410399864 usec\nrounds: 103660"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 286279.50625764474,
            "unit": "iter/sec",
            "range": "stddev: 3.246111487596104e-7",
            "extra": "mean: 3.493089718759064 usec\nrounds: 115394"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 284875.88052791264,
            "unit": "iter/sec",
            "range": "stddev: 3.2815878896010924e-7",
            "extra": "mean: 3.5103006900649785 usec\nrounds: 56520"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 286402.88662794145,
            "unit": "iter/sec",
            "range": "stddev: 3.698421613627101e-7",
            "extra": "mean: 3.4915849200188895 usec\nrounds: 47374"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 227188.89617344015,
            "unit": "iter/sec",
            "range": "stddev: 7.34658540250799e-7",
            "extra": "mean: 4.401623568946705 usec\nrounds: 101854"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 230549.04949454,
            "unit": "iter/sec",
            "range": "stddev: 8.566273583106373e-7",
            "extra": "mean: 4.337471796966496 usec\nrounds: 100929"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 223888.54666033926,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010687660246617625",
            "extra": "mean: 4.466508068039306 usec\nrounds: 70339"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 236536.21376880544,
            "unit": "iter/sec",
            "range": "stddev: 4.5788768471330306e-7",
            "extra": "mean: 4.227682451099928 usec\nrounds: 73828"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 94695.94125000613,
            "unit": "iter/sec",
            "range": "stddev: 4.857133200500568e-7",
            "extra": "mean: 10.560114687068863 usec\nrounds: 70191"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 99898.00992608744,
            "unit": "iter/sec",
            "range": "stddev: 9.970061115285545e-7",
            "extra": "mean: 10.010209419986246 usec\nrounds: 73121"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 94728.20871813473,
            "unit": "iter/sec",
            "range": "stddev: 6.837311739844642e-7",
            "extra": "mean: 10.556517573086552 usec\nrounds: 53434"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 97972.07604620619,
            "unit": "iter/sec",
            "range": "stddev: 7.904138957665771e-7",
            "extra": "mean: 10.206989995071389 usec\nrounds: 58371"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 88342.17302072505,
            "unit": "iter/sec",
            "range": "stddev: 6.419845467272731e-7",
            "extra": "mean: 11.319621940535697 usec\nrounds: 51029"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 94470.96123509703,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010801698434072448",
            "extra": "mean: 10.58526331188095 usec\nrounds: 66989"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 88115.24710902631,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013299144040885282",
            "extra": "mean: 11.348773711803647 usec\nrounds: 51293"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 97565.97400688214,
            "unit": "iter/sec",
            "range": "stddev: 0.000001571724192341733",
            "extra": "mean: 10.249474882805574 usec\nrounds: 54186"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2764.905896615325,
            "unit": "iter/sec",
            "range": "stddev: 0.00016629914724191299",
            "extra": "mean: 361.67596200078833 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3165.49146320048,
            "unit": "iter/sec",
            "range": "stddev: 0.000013841491626024743",
            "extra": "mean: 315.9067119988208 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2380.538467653844,
            "unit": "iter/sec",
            "range": "stddev: 0.0019373760283045874",
            "extra": "mean: 420.0730270011377 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3092.574042254321,
            "unit": "iter/sec",
            "range": "stddev: 0.00001374612375873266",
            "extra": "mean: 323.355232999063 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 10311.59176102587,
            "unit": "iter/sec",
            "range": "stddev: 0.00001073602157462669",
            "extra": "mean: 96.97823800391735 usec\nrounds: 7294"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 2431.389809836798,
            "unit": "iter/sec",
            "range": "stddev: 0.0000183886328023623",
            "extra": "mean: 411.2874027662075 usec\nrounds: 2242"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1356.2422151050869,
            "unit": "iter/sec",
            "range": "stddev: 0.00016466240306754794",
            "extra": "mean: 737.3314212332759 usec\nrounds: 876"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 533307.8684192922,
            "unit": "iter/sec",
            "range": "stddev: 2.165569067380918e-7",
            "extra": "mean: 1.875089529363159 usec\nrounds: 111025"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 94494.76084179372,
            "unit": "iter/sec",
            "range": "stddev: 7.07814596947847e-7",
            "extra": "mean: 10.58259728996228 usec\nrounds: 18671"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 74613.81916088877,
            "unit": "iter/sec",
            "range": "stddev: 9.880971275935712e-7",
            "extra": "mean: 13.402343041088857 usec\nrounds: 20808"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 60327.20425096738,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011340514703339317",
            "extra": "mean: 16.57626956886477 usec\nrounds: 20696"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 72242.89355707368,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027726854430804867",
            "extra": "mean: 13.842191954977206 usec\nrounds: 11013"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 86102.19877897372,
            "unit": "iter/sec",
            "range": "stddev: 8.942024373870602e-7",
            "extra": "mean: 11.614105263060965 usec\nrounds: 10184"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 42778.150944325454,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021440694623649784",
            "extra": "mean: 23.37641945537738 usec\nrounds: 10907"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 16819.556434598777,
            "unit": "iter/sec",
            "range": "stddev: 0.00001471855927259528",
            "extra": "mean: 59.4546000002083 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 51748.48603966264,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015154055128611039",
            "extra": "mean: 19.324236833393538 usec\nrounds: 14677"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 23774.244974527057,
            "unit": "iter/sec",
            "range": "stddev: 0.000004074879371359998",
            "extra": "mean: 42.06232421140823 usec\nrounds: 7199"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 27273.801117125473,
            "unit": "iter/sec",
            "range": "stddev: 0.000005516714167230505",
            "extra": "mean: 36.66522299937469 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 15979.129977973676,
            "unit": "iter/sec",
            "range": "stddev: 0.000003896048523632529",
            "extra": "mean: 62.581629999783665 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 11667.964455533616,
            "unit": "iter/sec",
            "range": "stddev: 0.000006800912718400052",
            "extra": "mean: 85.70475199945804 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 19460.021759688447,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034337246591720633",
            "extra": "mean: 51.38740399928565 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 48256.66519774088,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020172839794778182",
            "extra": "mean: 20.72252601588422 usec\nrounds: 16490"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 49434.23075885993,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021417906459808593",
            "extra": "mean: 20.228897762726355 usec\nrounds: 17567"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 6977.100867656798,
            "unit": "iter/sec",
            "range": "stddev: 0.0000689686366050877",
            "extra": "mean: 143.32600588241772 usec\nrounds: 3570"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 45322.513453342166,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023101503678426533",
            "extra": "mean: 22.064089650046935 usec\nrounds: 24116"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 33476.33620659685,
            "unit": "iter/sec",
            "range": "stddev: 0.000002478073431487983",
            "extra": "mean: 29.87184720061868 usec\nrounds: 14987"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 39184.35846594924,
            "unit": "iter/sec",
            "range": "stddev: 0.000002991118600365146",
            "extra": "mean: 25.520387194012343 usec\nrounds: 11245"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 45653.163968150446,
            "unit": "iter/sec",
            "range": "stddev: 0.000002086239560672148",
            "extra": "mean: 21.904286868214474 usec\nrounds: 14735"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 38161.66074994944,
            "unit": "iter/sec",
            "range": "stddev: 0.000002397795142550105",
            "extra": "mean: 26.20431030379947 usec\nrounds: 14344"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 23430.040827696746,
            "unit": "iter/sec",
            "range": "stddev: 0.000005320449568015113",
            "extra": "mean: 42.680249998450535 usec\nrounds: 828"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 16524.388460469698,
            "unit": "iter/sec",
            "range": "stddev: 0.000002428998071422406",
            "extra": "mean: 60.516611697445875 usec\nrounds: 11285"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 8225.714094667377,
            "unit": "iter/sec",
            "range": "stddev: 0.000004396616950602089",
            "extra": "mean: 121.56999240324762 usec\nrounds: 5397"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2514.2484555138644,
            "unit": "iter/sec",
            "range": "stddev: 0.000009736874377239302",
            "extra": "mean: 397.7331666673408 usec\nrounds: 2190"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 924107.4061999202,
            "unit": "iter/sec",
            "range": "stddev: 8.082596989511856e-8",
            "extra": "mean: 1.0821252954915044 usec\nrounds: 173612"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3807.530816240952,
            "unit": "iter/sec",
            "range": "stddev: 0.000050036434164675105",
            "extra": "mean: 262.6374015765069 usec\nrounds: 762"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 3877.9138624557436,
            "unit": "iter/sec",
            "range": "stddev: 0.000008628992184570192",
            "extra": "mean: 257.87060658607203 usec\nrounds: 3462"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1246.7749914712952,
            "unit": "iter/sec",
            "range": "stddev: 0.0034350124053703895",
            "extra": "mean: 802.0693443809932 usec\nrounds: 1388"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 618.6947550525751,
            "unit": "iter/sec",
            "range": "stddev: 0.0027149498460697814",
            "extra": "mean: 1.6163059276541345 msec\nrounds: 622"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 614.7820990919555,
            "unit": "iter/sec",
            "range": "stddev: 0.0029259165330046168",
            "extra": "mean: 1.626592578861711 msec\nrounds: 615"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 9334.87896343638,
            "unit": "iter/sec",
            "range": "stddev: 0.000010990781078764828",
            "extra": "mean: 107.12511687798866 usec\nrounds: 2524"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 11748.330665502494,
            "unit": "iter/sec",
            "range": "stddev: 0.0000445653473810466",
            "extra": "mean: 85.1184758474985 usec\nrounds: 8198"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 181362.4232860336,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015456097142788525",
            "extra": "mean: 5.513821341165373 usec\nrounds: 20917"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1492149.4837274444,
            "unit": "iter/sec",
            "range": "stddev: 1.4018098064523233e-7",
            "extra": "mean: 670.1741420048367 nsec\nrounds: 62271"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 280821.3025775993,
            "unit": "iter/sec",
            "range": "stddev: 4.0587964101116585e-7",
            "extra": "mean: 3.5609834112341603 usec\nrounds: 22847"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 4033.3071356382225,
            "unit": "iter/sec",
            "range": "stddev: 0.0000555529402375944",
            "extra": "mean: 247.9354947120242 usec\nrounds: 1702"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 3085.6040593108505,
            "unit": "iter/sec",
            "range": "stddev: 0.00005236828196979733",
            "extra": "mean: 324.0856509060153 usec\nrounds: 2707"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1637.9559963604008,
            "unit": "iter/sec",
            "range": "stddev: 0.00006102047807491251",
            "extra": "mean: 610.5170115815304 usec\nrounds: 1468"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 8.241728249452184,
            "unit": "iter/sec",
            "range": "stddev: 0.027576744725910093",
            "extra": "mean: 121.33377487500496 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 2.6062129926570745,
            "unit": "iter/sec",
            "range": "stddev: 0.08174299079358464",
            "extra": "mean: 383.6984938750092 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 8.50096436321098,
            "unit": "iter/sec",
            "range": "stddev: 0.030165459490173453",
            "extra": "mean: 117.63371275000623 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 7.790042619957322,
            "unit": "iter/sec",
            "range": "stddev: 0.004421976509838938",
            "extra": "mean: 128.36900242857448 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 2.4242825696271475,
            "unit": "iter/sec",
            "range": "stddev: 0.0893986190227175",
            "extra": "mean: 412.4931691249998 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_filesystem",
            "value": 10.578180366553463,
            "unit": "iter/sec",
            "range": "stddev: 0.008720992237355491",
            "extra": "mean: 94.53421716667283 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_filesystem",
            "value": 10.338396326529159,
            "unit": "iter/sec",
            "range": "stddev: 0.00418748780117732",
            "extra": "mean: 96.72680060000403 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 39495.39845713048,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016959250378052975",
            "extra": "mean: 25.319405274146828 usec\nrounds: 8873"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 29993.601063298334,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022931062832588593",
            "extra": "mean: 33.34044477985839 usec\nrounds: 7615"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 39283.80370115301,
            "unit": "iter/sec",
            "range": "stddev: 0.0000038077451101840564",
            "extra": "mean: 25.4557834472289 usec\nrounds: 13738"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 26481.365081297055,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021456374833528586",
            "extra": "mean: 37.76240374807068 usec\nrounds: 9818"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 33726.93862001515,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032192919087978014",
            "extra": "mean: 29.649889403438266 usec\nrounds: 10362"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 25926.744608785717,
            "unit": "iter/sec",
            "range": "stddev: 0.000004075426051709013",
            "extra": "mean: 38.570210610287454 usec\nrounds: 11894"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 35087.849790562446,
            "unit": "iter/sec",
            "range": "stddev: 0.000001945376817002589",
            "extra": "mean: 28.499894008009843 usec\nrounds: 7727"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 24820.043368548504,
            "unit": "iter/sec",
            "range": "stddev: 0.00000659721314910204",
            "extra": "mean: 40.29001823853303 usec\nrounds: 8992"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 28820.911732619716,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027130876986363936",
            "extra": "mean: 34.69702864632811 usec\nrounds: 7680"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 17702.911443046352,
            "unit": "iter/sec",
            "range": "stddev: 0.000004990399566949546",
            "extra": "mean: 56.487883544872886 usec\nrounds: 7007"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 9674.42337078017,
            "unit": "iter/sec",
            "range": "stddev: 0.000010121983334582536",
            "extra": "mean: 103.3653336921679 usec\nrounds: 3716"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 55388.736986668126,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016885310299808542",
            "extra": "mean: 18.054212000549796 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 28019.35128510647,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027639042347508784",
            "extra": "mean: 35.6896199995731 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 48345.22279315877,
            "unit": "iter/sec",
            "range": "stddev: 0.000001658572521018346",
            "extra": "mean: 20.684567000103016 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 24748.855420903845,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023243777639648083",
            "extra": "mean: 40.40590900035568 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 740.183660792985,
            "unit": "iter/sec",
            "range": "stddev: 0.000027718250943016686",
            "extra": "mean: 1.3510160423274737 msec\nrounds: 567"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 889.6570430623877,
            "unit": "iter/sec",
            "range": "stddev: 0.0025037213963252803",
            "extra": "mean: 1.1240286442939726 msec\nrounds: 894"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6223.321572760562,
            "unit": "iter/sec",
            "range": "stddev: 0.000007360555578295175",
            "extra": "mean: 160.6858955155063 usec\nrounds: 2431"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 6798.069922660124,
            "unit": "iter/sec",
            "range": "stddev: 0.000007166690772700535",
            "extra": "mean: 147.10057580706587 usec\nrounds: 3687"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 6898.649318985703,
            "unit": "iter/sec",
            "range": "stddev: 0.000007602916619271651",
            "extra": "mean: 144.95591147790483 usec\nrounds: 2056"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4514.606027571777,
            "unit": "iter/sec",
            "range": "stddev: 0.00001337081901084577",
            "extra": "mean: 221.50327047205477 usec\nrounds: 2418"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 1974.8909796332864,
            "unit": "iter/sec",
            "range": "stddev: 0.00006065131592659032",
            "extra": "mean: 506.35706492805394 usec\nrounds: 1802"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 2054.152824106069,
            "unit": "iter/sec",
            "range": "stddev: 0.000020171903418978034",
            "extra": "mean: 486.8186963816493 usec\nrounds: 1907"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 2529.6063655029143,
            "unit": "iter/sec",
            "range": "stddev: 0.000018613270142866575",
            "extra": "mean: 395.31842330780535 usec\nrounds: 2334"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 2528.466664436979,
            "unit": "iter/sec",
            "range": "stddev: 0.000015283365356590508",
            "extra": "mean: 395.4966122611202 usec\nrounds: 2414"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1767.6423640658159,
            "unit": "iter/sec",
            "range": "stddev: 0.00006089710839233927",
            "extra": "mean: 565.7252962074664 usec\nrounds: 1688"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1704.4181899751668,
            "unit": "iter/sec",
            "range": "stddev: 0.00008151045253330148",
            "extra": "mean: 586.7104715742149 usec\nrounds: 1671"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 2189.4875143451745,
            "unit": "iter/sec",
            "range": "stddev: 0.000018092247535874274",
            "extra": "mean: 456.72788424147603 usec\nrounds: 2056"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 2183.275285199448,
            "unit": "iter/sec",
            "range": "stddev: 0.000024291924576978826",
            "extra": "mean: 458.0274447198936 usec\nrounds: 2017"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 584.8418099706981,
            "unit": "iter/sec",
            "range": "stddev: 0.000022002243836221377",
            "extra": "mean: 1.7098640742701043 msec\nrounds: 377"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 600.5772598769519,
            "unit": "iter/sec",
            "range": "stddev: 0.00003427604252437547",
            "extra": "mean: 1.6650647082523289 msec\nrounds: 521"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6d63bd4b6a7e7213865502a262160f1e0a703374",
          "message": "[Doc,CI] Sanitize version (#762)",
          "timestamp": "2024-04-25T16:35:14+01:00",
          "tree_id": "b8e62acdd88cbad04425d07d3b92929c2c161af0",
          "url": "https://github.com/pytorch/tensordict/commit/6d63bd4b6a7e7213865502a262160f1e0a703374"
        },
        "date": 1714059589256,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 56707.36799668245,
            "unit": "iter/sec",
            "range": "stddev: 9.366474049795301e-7",
            "extra": "mean: 17.6343927663598 usec\nrounds: 13051"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 54866.37212866574,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010514462415033948",
            "extra": "mean: 18.22610027240229 usec\nrounds: 17263"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 48586.90490798911,
            "unit": "iter/sec",
            "range": "stddev: 0.000001354428475036593",
            "extra": "mean: 20.58167734482652 usec\nrounds: 19510"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 48896.35196037858,
            "unit": "iter/sec",
            "range": "stddev: 0.000001219297826799205",
            "extra": "mean: 20.451423468366606 usec\nrounds: 33117"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 400627.85766104964,
            "unit": "iter/sec",
            "range": "stddev: 2.1051104212518482e-7",
            "extra": "mean: 2.4960820394223506 usec\nrounds: 158680"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 3769.5285207080215,
            "unit": "iter/sec",
            "range": "stddev: 0.000013049095017321754",
            "extra": "mean: 265.28516617037627 usec\nrounds: 2353"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 3743.0052019627888,
            "unit": "iter/sec",
            "range": "stddev: 0.0000048513374373048215",
            "extra": "mean: 267.16500406561323 usec\nrounds: 3689"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 12249.910758786247,
            "unit": "iter/sec",
            "range": "stddev: 0.000025526805015502673",
            "extra": "mean: 81.63324775919287 usec\nrounds: 9929"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 3700.559319131403,
            "unit": "iter/sec",
            "range": "stddev: 0.000010907904418651537",
            "extra": "mean: 270.2294204095397 usec\nrounds: 3518"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 13036.628732413423,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022173146476159414",
            "extra": "mean: 76.70694782568022 usec\nrounds: 11155"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 3739.615999196678,
            "unit": "iter/sec",
            "range": "stddev: 0.000011270926132878128",
            "extra": "mean: 267.40713490765205 usec\nrounds: 3558"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 255033.00873859535,
            "unit": "iter/sec",
            "range": "stddev: 2.611315747775196e-7",
            "extra": "mean: 3.921061061648626 usec\nrounds: 118274"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 7208.633859076595,
            "unit": "iter/sec",
            "range": "stddev: 0.0000056680790731425385",
            "extra": "mean: 138.72254015799007 usec\nrounds: 6076"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 6928.926185458946,
            "unit": "iter/sec",
            "range": "stddev: 0.00002354115870366424",
            "extra": "mean: 144.32250730258914 usec\nrounds: 6436"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 8526.725644897277,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033559173003006963",
            "extra": "mean: 117.27831311172055 usec\nrounds: 7665"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 7177.739370417948,
            "unit": "iter/sec",
            "range": "stddev: 0.000006066620770499334",
            "extra": "mean: 139.3196309302286 usec\nrounds: 6492"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 8487.260944267968,
            "unit": "iter/sec",
            "range": "stddev: 0.000005193778142115357",
            "extra": "mean: 117.82364258228314 usec\nrounds: 7543"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 6997.575752136853,
            "unit": "iter/sec",
            "range": "stddev: 0.000005677247367946267",
            "extra": "mean: 142.90663444331125 usec\nrounds: 6486"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 848390.7712385578,
            "unit": "iter/sec",
            "range": "stddev: 2.8677626337816376e-7",
            "extra": "mean: 1.1787021192369989 usec\nrounds: 194175"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 19753.535652382798,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014432923985896933",
            "extra": "mean: 50.62384869208837 usec\nrounds: 15789"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 19617.350030537626,
            "unit": "iter/sec",
            "range": "stddev: 0.000002617426461979352",
            "extra": "mean: 50.97528455389417 usec\nrounds: 17726"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 21482.91628921999,
            "unit": "iter/sec",
            "range": "stddev: 0.000001363523922811454",
            "extra": "mean: 46.548615026805955 usec\nrounds: 17635"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 18541.533082394086,
            "unit": "iter/sec",
            "range": "stddev: 0.000002483925619140383",
            "extra": "mean: 53.93297283219472 usec\nrounds: 15533"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 21233.54184821698,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014595439855519222",
            "extra": "mean: 47.09529889776593 usec\nrounds: 17598"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 18278.573797835623,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015966592352396196",
            "extra": "mean: 54.70886356124845 usec\nrounds: 16315"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 741577.6007503307,
            "unit": "iter/sec",
            "range": "stddev: 1.6487997376682974e-7",
            "extra": "mean: 1.3484765437739712 usec\nrounds: 172385"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 289332.5537274462,
            "unit": "iter/sec",
            "range": "stddev: 2.633763733044975e-7",
            "extra": "mean: 3.456230510936591 usec\nrounds: 112524"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 275399.16351354786,
            "unit": "iter/sec",
            "range": "stddev: 7.882752942902256e-7",
            "extra": "mean: 3.631093091358669 usec\nrounds: 122310"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 282987.0856985082,
            "unit": "iter/sec",
            "range": "stddev: 8.095363395088519e-7",
            "extra": "mean: 3.5337301613310745 usec\nrounds: 51238"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 288850.93704309314,
            "unit": "iter/sec",
            "range": "stddev: 2.426838120381298e-7",
            "extra": "mean: 3.4619932697355655 usec\nrounds: 53491"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 235172.01727236924,
            "unit": "iter/sec",
            "range": "stddev: 3.262427018640142e-7",
            "extra": "mean: 4.252206583072466 usec\nrounds: 109099"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 235186.434900118,
            "unit": "iter/sec",
            "range": "stddev: 3.202239437157313e-7",
            "extra": "mean: 4.251945910165664 usec\nrounds: 110169"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 233304.1665718015,
            "unit": "iter/sec",
            "range": "stddev: 3.183284862894472e-7",
            "extra": "mean: 4.286250068715515 usec\nrounds: 76139"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 236580.71415388884,
            "unit": "iter/sec",
            "range": "stddev: 3.501716457134597e-7",
            "extra": "mean: 4.226887232023187 usec\nrounds: 77256"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 96243.50734957124,
            "unit": "iter/sec",
            "range": "stddev: 4.6320627871183736e-7",
            "extra": "mean: 10.390311279573863 usec\nrounds: 69799"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 101109.56357111021,
            "unit": "iter/sec",
            "range": "stddev: 7.165928840193906e-7",
            "extra": "mean: 9.890261263927833 usec\nrounds: 75506"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 95286.63683948922,
            "unit": "iter/sec",
            "range": "stddev: 7.197318876832163e-7",
            "extra": "mean: 10.494651014753567 usec\nrounds: 55733"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 101719.95421180938,
            "unit": "iter/sec",
            "range": "stddev: 7.796508941265616e-7",
            "extra": "mean: 9.830912801216176 usec\nrounds: 60861"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 90452.5092216236,
            "unit": "iter/sec",
            "range": "stddev: 5.383178279235611e-7",
            "extra": "mean: 11.0555252541401 usec\nrounds: 58545"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 97934.96926869298,
            "unit": "iter/sec",
            "range": "stddev: 7.443247754561351e-7",
            "extra": "mean: 10.210857342043111 usec\nrounds: 70294"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 89866.153812967,
            "unit": "iter/sec",
            "range": "stddev: 5.173311784912996e-7",
            "extra": "mean: 11.12765994282163 usec\nrounds: 53491"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 98527.10640068336,
            "unit": "iter/sec",
            "range": "stddev: 7.377723508707801e-7",
            "extra": "mean: 10.149491206341407 usec\nrounds: 59134"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2836.489397228298,
            "unit": "iter/sec",
            "range": "stddev: 0.00016766390686168153",
            "extra": "mean: 352.5484709998068 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3303.53796461425,
            "unit": "iter/sec",
            "range": "stddev: 0.000015772193870972116",
            "extra": "mean: 302.7057690002266 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2524.1254076514674,
            "unit": "iter/sec",
            "range": "stddev: 0.0014594617965024402",
            "extra": "mean: 396.17682899933016 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3213.309827508503,
            "unit": "iter/sec",
            "range": "stddev: 0.000011756395446933912",
            "extra": "mean: 311.20559599924036 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 10326.343744181595,
            "unit": "iter/sec",
            "range": "stddev: 0.000006175169379760373",
            "extra": "mean: 96.83969706736255 usec\nrounds: 7774"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 2464.8927257558057,
            "unit": "iter/sec",
            "range": "stddev: 0.000008196602377487672",
            "extra": "mean: 405.697168704724 usec\nrounds: 2288"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1365.2460930611664,
            "unit": "iter/sec",
            "range": "stddev: 0.00010589730518434976",
            "extra": "mean: 732.4686773194065 usec\nrounds: 970"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 518135.50133491773,
            "unit": "iter/sec",
            "range": "stddev: 4.396209307254393e-7",
            "extra": "mean: 1.9299970710820096 usec\nrounds: 134518"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 84693.00081279328,
            "unit": "iter/sec",
            "range": "stddev: 9.657643389640132e-7",
            "extra": "mean: 11.80735114357815 usec\nrounds: 20157"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 68675.44225734727,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016390780946578137",
            "extra": "mean: 14.561245870870453 usec\nrounds: 25973"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 56532.681220495,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012644793795352052",
            "extra": "mean: 17.688883286814043 usec\nrounds: 23365"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 74965.04250030071,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016750362895320744",
            "extra": "mean: 13.339550897953385 usec\nrounds: 15197"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 85510.64212443562,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014590247696745682",
            "extra": "mean: 11.694450832737214 usec\nrounds: 11827"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 43939.66117587748,
            "unit": "iter/sec",
            "range": "stddev: 0.000002389721573758661",
            "extra": "mean: 22.758482274073426 usec\nrounds: 12242"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 17127.68689926701,
            "unit": "iter/sec",
            "range": "stddev: 0.000012311868386505256",
            "extra": "mean: 58.38500002255387 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 53207.33513879814,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013628687925111208",
            "extra": "mean: 18.794401136448048 usec\nrounds: 16017"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 25829.727360788434,
            "unit": "iter/sec",
            "range": "stddev: 0.000004299441296673335",
            "extra": "mean: 38.715081504037826 usec\nrounds: 10797"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 26895.296770455487,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029749193616394806",
            "extra": "mean: 37.181222000810976 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 15900.877744348987,
            "unit": "iter/sec",
            "range": "stddev: 0.000003915915802565833",
            "extra": "mean: 62.88961000001336 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 11789.113786428417,
            "unit": "iter/sec",
            "range": "stddev: 0.000003988445459515291",
            "extra": "mean: 84.8240179979598 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 19120.793778823558,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034909941091157126",
            "extra": "mean: 52.299084000765106 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 47859.26314079022,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017955898578536055",
            "extra": "mean: 20.89459666477198 usec\nrounds: 19009"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 49107.633211699605,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017238842002946946",
            "extra": "mean: 20.36343302657388 usec\nrounds: 19560"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 7102.758076852691,
            "unit": "iter/sec",
            "range": "stddev: 0.00009136772383981973",
            "extra": "mean: 140.79037877679073 usec\nrounds: 4137"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 42710.248664867446,
            "unit": "iter/sec",
            "range": "stddev: 0.00000432638610648941",
            "extra": "mean: 23.413584122318138 usec\nrounds: 17748"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 31399.415620440064,
            "unit": "iter/sec",
            "range": "stddev: 0.000004726991532606518",
            "extra": "mean: 31.84772647007578 usec\nrounds: 18689"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 38239.85986092277,
            "unit": "iter/sec",
            "range": "stddev: 0.000009235509983403578",
            "extra": "mean: 26.150723450268128 usec\nrounds: 12486"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 45220.923251870605,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020309484017138574",
            "extra": "mean: 22.11365730925527 usec\nrounds: 17882"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 38383.388035218806,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018629069501746692",
            "extra": "mean: 26.05293725198116 usec\nrounds: 15889"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 23636.914175504662,
            "unit": "iter/sec",
            "range": "stddev: 0.0000055597677767705566",
            "extra": "mean: 42.30670689815835 usec\nrounds: 11641"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 16502.44621779922,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037066002200095856",
            "extra": "mean: 60.59707674862284 usec\nrounds: 11440"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 8261.226353587605,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034773637360957813",
            "extra": "mean: 121.04740352087433 usec\nrounds: 5737"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2518.164440989919,
            "unit": "iter/sec",
            "range": "stddev: 0.000006730303545889266",
            "extra": "mean: 397.1146537224903 usec\nrounds: 2189"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 913744.6206385248,
            "unit": "iter/sec",
            "range": "stddev: 1.2455073174472197e-7",
            "extra": "mean: 1.0943976877272352 usec\nrounds: 179824"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3796.568660125309,
            "unit": "iter/sec",
            "range": "stddev: 0.00004825115392301812",
            "extra": "mean: 263.3957369197148 usec\nrounds: 688"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 4005.4348273426745,
            "unit": "iter/sec",
            "range": "stddev: 0.000014675775688524818",
            "extra": "mean: 249.66078418592818 usec\nrounds: 3554"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1369.8980143618994,
            "unit": "iter/sec",
            "range": "stddev: 0.0024131966935921675",
            "extra": "mean: 729.9813486230955 usec\nrounds: 1526"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 623.7079859041999,
            "unit": "iter/sec",
            "range": "stddev: 0.002406155410479953",
            "extra": "mean: 1.603314407703604 msec\nrounds: 623"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 629.2206243359128,
            "unit": "iter/sec",
            "range": "stddev: 0.002054175400937642",
            "extra": "mean: 1.5892676770654368 msec\nrounds: 641"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 9491.871812307665,
            "unit": "iter/sec",
            "range": "stddev: 0.00007295669939014177",
            "extra": "mean: 105.3532980400501 usec\nrounds: 2805"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 12207.54583408609,
            "unit": "iter/sec",
            "range": "stddev: 0.000006638715845901616",
            "extra": "mean: 81.91654683022244 usec\nrounds: 9118"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 179369.4842103543,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019163542785328383",
            "extra": "mean: 5.575084326090034 usec\nrounds: 25022"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1554135.9346938517,
            "unit": "iter/sec",
            "range": "stddev: 1.0657257625046248e-7",
            "extra": "mean: 643.444358808285 nsec\nrounds: 75160"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 283997.55260627135,
            "unit": "iter/sec",
            "range": "stddev: 3.87896630938619e-7",
            "extra": "mean: 3.521157104428926 usec\nrounds: 30903"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 4057.6485357581028,
            "unit": "iter/sec",
            "range": "stddev: 0.00005044648717215196",
            "extra": "mean: 246.44815616421224 usec\nrounds: 1825"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 3118.9781693183436,
            "unit": "iter/sec",
            "range": "stddev: 0.000050505966527474997",
            "extra": "mean: 320.6178260037489 usec\nrounds: 2615"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1543.7042603939244,
            "unit": "iter/sec",
            "range": "stddev: 0.00005703044971348761",
            "extra": "mean: 647.7924727271393 usec\nrounds: 1430"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 9.092572856090749,
            "unit": "iter/sec",
            "range": "stddev: 0.023721848635556472",
            "extra": "mean: 109.97987212498828 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 2.5707860012641444,
            "unit": "iter/sec",
            "range": "stddev: 0.12604495502419624",
            "extra": "mean: 388.9860919999819 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.234609805537751,
            "unit": "iter/sec",
            "range": "stddev: 0.020099599014642335",
            "extra": "mean: 108.28827866666619 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 7.308853037418818,
            "unit": "iter/sec",
            "range": "stddev: 0.03996233238201594",
            "extra": "mean: 136.82037316667106 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 2.4613469330440085,
            "unit": "iter/sec",
            "range": "stddev: 0.10024221024081272",
            "extra": "mean: 406.2816121428585 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_filesystem",
            "value": 11.06027935198928,
            "unit": "iter/sec",
            "range": "stddev: 0.0035173044686459946",
            "extra": "mean: 90.41362954545465 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_filesystem",
            "value": 10.345789999499221,
            "unit": "iter/sec",
            "range": "stddev: 0.0027539357321100613",
            "extra": "mean: 96.65767428571469 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 39006.683188359595,
            "unit": "iter/sec",
            "range": "stddev: 0.000001618243433211223",
            "extra": "mean: 25.636632450164868 usec\nrounds: 10453"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 30152.610190129417,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017575703113384569",
            "extra": "mean: 33.16462467741364 usec\nrounds: 8518"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 38991.20955585369,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014852122249870834",
            "extra": "mean: 25.64680632868112 usec\nrounds: 17824"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 26430.613416568096,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031325558860198694",
            "extra": "mean: 37.83491454546217 usec\nrounds: 11866"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 34163.143116736115,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014439085212503198",
            "extra": "mean: 29.271311383234874 usec\nrounds: 13151"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 26126.544280196373,
            "unit": "iter/sec",
            "range": "stddev: 0.000004462065453594056",
            "extra": "mean: 38.27524946565508 usec\nrounds: 12627"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 34295.655578566635,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018580458695878545",
            "extra": "mean: 29.158212115500675 usec\nrounds: 10169"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 24487.453883438735,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020451394112915047",
            "extra": "mean: 40.837238724778835 usec\nrounds: 11574"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 27890.525492345845,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036334657582096505",
            "extra": "mean: 35.854469657605975 usec\nrounds: 10365"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 17008.44594163749,
            "unit": "iter/sec",
            "range": "stddev: 0.000003525222682587104",
            "extra": "mean: 58.79431921243035 usec\nrounds: 8380"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 9902.404029910776,
            "unit": "iter/sec",
            "range": "stddev: 0.000011559881016086061",
            "extra": "mean: 100.98557855036444 usec\nrounds: 4233"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 55458.66925980537,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023662238247692744",
            "extra": "mean: 18.031446000179585 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 27853.105728589584,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022487551670314665",
            "extra": "mean: 35.902638999914416 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 48911.740673697226,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011027443762716217",
            "extra": "mean: 20.44498899908831 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 24556.455167266642,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021175920517087497",
            "extra": "mean: 40.72249000063266 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 769.4189868215199,
            "unit": "iter/sec",
            "range": "stddev: 0.000018977329837043117",
            "extra": "mean: 1.2996819900832102 msec\nrounds: 605"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 990.4469663068803,
            "unit": "iter/sec",
            "range": "stddev: 0.000093380247079707",
            "extra": "mean: 1.0096451743688413 msec\nrounds: 952"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6402.285560349838,
            "unit": "iter/sec",
            "range": "stddev: 0.000005311121665804175",
            "extra": "mean: 156.1942201068203 usec\nrounds: 2626"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 6897.152720001946,
            "unit": "iter/sec",
            "range": "stddev: 0.000005709385738970125",
            "extra": "mean: 144.98736516301437 usec\nrounds: 4168"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 7140.874892471241,
            "unit": "iter/sec",
            "range": "stddev: 0.000007102665387153299",
            "extra": "mean: 140.0388628926014 usec\nrounds: 2261"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4674.172558881526,
            "unit": "iter/sec",
            "range": "stddev: 0.000017201972082240823",
            "extra": "mean: 213.94160942985985 usec\nrounds: 2842"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 2099.7066926263606,
            "unit": "iter/sec",
            "range": "stddev: 0.000017296548522715013",
            "extra": "mean: 476.2569950897177 usec\nrounds: 1833"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 2118.105804154747,
            "unit": "iter/sec",
            "range": "stddev: 0.000018812352696919317",
            "extra": "mean: 472.1199470009765 usec\nrounds: 2000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 2576.3281319653297,
            "unit": "iter/sec",
            "range": "stddev: 0.000013848717520397381",
            "extra": "mean: 388.14931514067604 usec\nrounds: 2272"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 2579.023838342169,
            "unit": "iter/sec",
            "range": "stddev: 0.000015268605267679733",
            "extra": "mean: 387.7436048217427 usec\nrounds: 1784"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1836.0049968308838,
            "unit": "iter/sec",
            "range": "stddev: 0.000026716496297973756",
            "extra": "mean: 544.6608270272105 usec\nrounds: 1665"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1828.4719917846187,
            "unit": "iter/sec",
            "range": "stddev: 0.00004711277963524172",
            "extra": "mean: 546.9047404023858 usec\nrounds: 1745"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 2238.1102611207743,
            "unit": "iter/sec",
            "range": "stddev: 0.000019280586560998674",
            "extra": "mean: 446.80551149398326 usec\nrounds: 2088"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 2248.1897740779596,
            "unit": "iter/sec",
            "range": "stddev: 0.000019121665811219555",
            "extra": "mean: 444.8023078523812 usec\nrounds: 1936"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 592.0822093006287,
            "unit": "iter/sec",
            "range": "stddev: 0.00003282701816178439",
            "extra": "mean: 1.6889546490194434 msec\nrounds: 510"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 604.0335035766261,
            "unit": "iter/sec",
            "range": "stddev: 0.00002596371500918728",
            "extra": "mean: 1.6555373072499489 msec\nrounds: 524"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7dbb8649f13b17be973918ec3a2dae10c985b5c4",
          "message": "[CI] Fix wheels (#763)",
          "timestamp": "2024-04-25T16:49:14+01:00",
          "tree_id": "1ea4255cf91c241f088b0c90523819fe203b0700",
          "url": "https://github.com/pytorch/tensordict/commit/7dbb8649f13b17be973918ec3a2dae10c985b5c4"
        },
        "date": 1714060416784,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 63923.233221484275,
            "unit": "iter/sec",
            "range": "stddev: 8.433989018390908e-7",
            "extra": "mean: 15.64376439682192 usec\nrounds: 8943"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 62083.25974174578,
            "unit": "iter/sec",
            "range": "stddev: 8.579784793285574e-7",
            "extra": "mean: 16.107401643531677 usec\nrounds: 18011"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 54912.266145988324,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011488628135308203",
            "extra": "mean: 18.210867447018593 usec\nrounds: 31527"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 55301.13043595861,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010242608202681637",
            "extra": "mean: 18.082812993453153 usec\nrounds: 29400"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 391772.5667509209,
            "unit": "iter/sec",
            "range": "stddev: 1.764551800010766e-7",
            "extra": "mean: 2.5525013358982194 usec\nrounds: 119389"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 3762.3900952752333,
            "unit": "iter/sec",
            "range": "stddev: 0.00002025490274006325",
            "extra": "mean: 265.78849472727154 usec\nrounds: 3319"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 3723.0179231687302,
            "unit": "iter/sec",
            "range": "stddev: 0.000032077816358034176",
            "extra": "mean: 268.5992978376213 usec\nrounds: 3653"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 12485.674485827518,
            "unit": "iter/sec",
            "range": "stddev: 0.000026340401289033326",
            "extra": "mean: 80.09178848408224 usec\nrounds: 10212"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 3809.686827347007,
            "unit": "iter/sec",
            "range": "stddev: 0.0000048088358391514255",
            "extra": "mean: 262.48876753378204 usec\nrounds: 3536"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 12736.752425205135,
            "unit": "iter/sec",
            "range": "stddev: 0.000002071645774874588",
            "extra": "mean: 78.51294950360112 usec\nrounds: 10872"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 3774.403560682127,
            "unit": "iter/sec",
            "range": "stddev: 0.000005717022908163447",
            "extra": "mean: 264.9425224204895 usec\nrounds: 3702"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 258820.4697396032,
            "unit": "iter/sec",
            "range": "stddev: 4.0305785964316014e-7",
            "extra": "mean: 3.863682038001439 usec\nrounds: 117014"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 7211.695500565314,
            "unit": "iter/sec",
            "range": "stddev: 0.000005311517555018617",
            "extra": "mean: 138.6636471162172 usec\nrounds: 6121"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 6995.497065934337,
            "unit": "iter/sec",
            "range": "stddev: 0.000022931824741218158",
            "extra": "mean: 142.9490986236926 usec\nrounds: 6540"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 8544.116985181281,
            "unit": "iter/sec",
            "range": "stddev: 0.00000316997991366547",
            "extra": "mean: 117.03959598567963 usec\nrounds: 7673"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 7065.246729699753,
            "unit": "iter/sec",
            "range": "stddev: 0.000016897450280359593",
            "extra": "mean: 141.53787380083415 usec\nrounds: 6569"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 8562.135954188127,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026813042093018275",
            "extra": "mean: 116.7932867862084 usec\nrounds: 7598"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 6978.239123436045,
            "unit": "iter/sec",
            "range": "stddev: 0.0000063089793209068144",
            "extra": "mean: 143.3026272547115 usec\nrounds: 6487"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 868741.8581001417,
            "unit": "iter/sec",
            "range": "stddev: 6.025714581326338e-8",
            "extra": "mean: 1.1510899246721016 usec\nrounds: 162049"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 19705.094423669423,
            "unit": "iter/sec",
            "range": "stddev: 0.000005411785358092772",
            "extra": "mean: 50.748297800533095 usec\nrounds: 15685"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 19529.20850165701,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018009087603925265",
            "extra": "mean: 51.20535222485603 usec\nrounds: 16180"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 21718.26094718728,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016749717655031746",
            "extra": "mean: 46.04420226977287 usec\nrounds: 17180"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 19387.68121827857,
            "unit": "iter/sec",
            "range": "stddev: 0.00000179626201957103",
            "extra": "mean: 51.57914392863067 usec\nrounds: 16265"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 21785.78082075901,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018647919927536267",
            "extra": "mean: 45.9014991579797 usec\nrounds: 17812"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 19503.016491217855,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014915652210009074",
            "extra": "mean: 51.274119593258646 usec\nrounds: 17802"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 725218.9875490336,
            "unit": "iter/sec",
            "range": "stddev: 1.5101415829606e-7",
            "extra": "mean: 1.3788938474702965 usec\nrounds: 148965"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 289496.7684405763,
            "unit": "iter/sec",
            "range": "stddev: 2.8182798591322177e-7",
            "extra": "mean: 3.4542699919818465 usec\nrounds: 111533"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 287914.46232964384,
            "unit": "iter/sec",
            "range": "stddev: 2.3346875188700936e-7",
            "extra": "mean: 3.473253798744793 usec\nrounds: 119977"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 288144.64213394606,
            "unit": "iter/sec",
            "range": "stddev: 2.623974288415552e-7",
            "extra": "mean: 3.4704792447091313 usec\nrounds: 86267"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 280350.85199495894,
            "unit": "iter/sec",
            "range": "stddev: 7.353171274205157e-7",
            "extra": "mean: 3.5669590189723457 usec\nrounds: 53781"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 215049.45461253618,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015832493738079483",
            "extra": "mean: 4.650093169507185 usec\nrounds: 60202"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 238148.16460538172,
            "unit": "iter/sec",
            "range": "stddev: 2.9992047381270966e-7",
            "extra": "mean: 4.199066583851395 usec\nrounds: 90172"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 241086.10139530196,
            "unit": "iter/sec",
            "range": "stddev: 2.750800903380588e-7",
            "extra": "mean: 4.147895686281512 usec\nrounds: 78101"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 242156.1044595442,
            "unit": "iter/sec",
            "range": "stddev: 3.0727601008698347e-7",
            "extra": "mean: 4.129567587122566 usec\nrounds: 78469"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 95063.34308487223,
            "unit": "iter/sec",
            "range": "stddev: 5.823101438158289e-7",
            "extra": "mean: 10.519301841796194 usec\nrounds: 72485"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 98835.18697050955,
            "unit": "iter/sec",
            "range": "stddev: 7.382427490845107e-7",
            "extra": "mean: 10.117854082659653 usec\nrounds: 76605"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 95474.59230284452,
            "unit": "iter/sec",
            "range": "stddev: 4.748534426566869e-7",
            "extra": "mean: 10.473990785192454 usec\nrounds: 58167"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 100239.24968243465,
            "unit": "iter/sec",
            "range": "stddev: 7.056957973733044e-7",
            "extra": "mean: 9.976132135546445 usec\nrounds: 61611"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 90403.86846793156,
            "unit": "iter/sec",
            "range": "stddev: 5.069555989424894e-7",
            "extra": "mean: 11.061473551374897 usec\nrounds: 53519"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 97699.00100074074,
            "unit": "iter/sec",
            "range": "stddev: 8.216219899126276e-7",
            "extra": "mean: 10.23551919422818 usec\nrounds: 69604"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 91132.89898779239,
            "unit": "iter/sec",
            "range": "stddev: 5.428422140669069e-7",
            "extra": "mean: 10.972985728611068 usec\nrounds: 54514"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 98702.33486117155,
            "unit": "iter/sec",
            "range": "stddev: 7.579915498568586e-7",
            "extra": "mean: 10.13147258782213 usec\nrounds: 59517"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2862.973425225445,
            "unit": "iter/sec",
            "range": "stddev: 0.0001376023702931543",
            "extra": "mean: 349.28720999960206 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3222.9620316730434,
            "unit": "iter/sec",
            "range": "stddev: 0.000008386627417869125",
            "extra": "mean: 310.2735899997242 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2511.409944395887,
            "unit": "iter/sec",
            "range": "stddev: 0.0015643115720475743",
            "extra": "mean: 398.18270299974756 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3117.948097800333,
            "unit": "iter/sec",
            "range": "stddev: 0.000018851025139854983",
            "extra": "mean: 320.723748001285 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 10400.86446518948,
            "unit": "iter/sec",
            "range": "stddev: 0.00000903544523131086",
            "extra": "mean: 96.14585435151926 usec\nrounds: 7676"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 2454.1344850600804,
            "unit": "iter/sec",
            "range": "stddev: 0.000012701464900591154",
            "extra": "mean: 407.4756318725209 usec\nrounds: 2328"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1484.1290774914664,
            "unit": "iter/sec",
            "range": "stddev: 0.0001615835672844983",
            "extra": "mean: 673.7958410532859 usec\nrounds: 950"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 540119.0053340268,
            "unit": "iter/sec",
            "range": "stddev: 1.7358270365907407e-7",
            "extra": "mean: 1.8514438302010279 usec\nrounds: 125708"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 117081.30778217995,
            "unit": "iter/sec",
            "range": "stddev: 4.0482473527079656e-7",
            "extra": "mean: 8.541073028159346 usec\nrounds: 24292"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 88202.41044092405,
            "unit": "iter/sec",
            "range": "stddev: 7.660888792486706e-7",
            "extra": "mean: 11.337558633613272 usec\nrounds: 26171"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 68543.3352796404,
            "unit": "iter/sec",
            "range": "stddev: 7.446095424468468e-7",
            "extra": "mean: 14.589310483947699 usec\nrounds: 25048"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 76433.33089109974,
            "unit": "iter/sec",
            "range": "stddev: 0.000001291349873393638",
            "extra": "mean: 13.083297408885326 usec\nrounds: 14781"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 87600.30504683136,
            "unit": "iter/sec",
            "range": "stddev: 7.925498791384148e-7",
            "extra": "mean: 11.415485362355728 usec\nrounds: 11853"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 44026.04559535305,
            "unit": "iter/sec",
            "range": "stddev: 0.000002551867185876116",
            "extra": "mean: 22.713827382797014 usec\nrounds: 11343"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 17540.105454637032,
            "unit": "iter/sec",
            "range": "stddev: 0.000011009804495509238",
            "extra": "mean: 57.01219998854867 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 53178.62405243065,
            "unit": "iter/sec",
            "range": "stddev: 0.000001193131111525691",
            "extra": "mean: 18.804548214975725 usec\nrounds: 16105"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 24358.93527056086,
            "unit": "iter/sec",
            "range": "stddev: 0.0000047339484692706",
            "extra": "mean: 41.05269745548181 usec\nrounds: 7860"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 30865.882769793094,
            "unit": "iter/sec",
            "range": "stddev: 0.000001692420940411981",
            "extra": "mean: 32.39823100017247 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 17014.474860163624,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026143919340865842",
            "extra": "mean: 58.77348600051846 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 12075.49896586913,
            "unit": "iter/sec",
            "range": "stddev: 0.00000528307505264525",
            "extra": "mean: 82.81231300060199 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 21264.0779885884,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018790533747118804",
            "extra": "mean: 47.02766800124891 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 54597.253996904416,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016238377175520047",
            "extra": "mean: 18.31593948033904 usec\nrounds: 18275"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 56179.839553129554,
            "unit": "iter/sec",
            "range": "stddev: 0.000001819580017677227",
            "extra": "mean: 17.79997963600973 usec\nrounds: 20821"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 7185.133576873459,
            "unit": "iter/sec",
            "range": "stddev: 0.00006650928490566015",
            "extra": "mean: 139.17625737935694 usec\nrounds: 3625"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 52850.38284759189,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019097169184146504",
            "extra": "mean: 18.921338808155195 usec\nrounds: 25206"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 36754.59706712604,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026253864684567082",
            "extra": "mean: 27.20748096282132 usec\nrounds: 19199"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 41054.638824536,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032593401766713692",
            "extra": "mean: 24.357783398702257 usec\nrounds: 11662"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 51270.54285765036,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016388686096868586",
            "extra": "mean: 19.504377060653347 usec\nrounds: 18440"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 42470.62686816081,
            "unit": "iter/sec",
            "range": "stddev: 0.000002006271409140109",
            "extra": "mean: 23.545684953138178 usec\nrounds: 16515"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 25998.807012503024,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025073828787832533",
            "extra": "mean: 38.463303316921134 usec\nrounds: 12060"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 16593.417854699208,
            "unit": "iter/sec",
            "range": "stddev: 0.000002742531585543198",
            "extra": "mean: 60.2648597628609 usec\nrounds: 927"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 8367.025466513991,
            "unit": "iter/sec",
            "range": "stddev: 0.000007860858561654403",
            "extra": "mean: 119.5167869396526 usec\nrounds: 5421"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2567.0492093617086,
            "unit": "iter/sec",
            "range": "stddev: 0.00002636343290043719",
            "extra": "mean: 389.552329715038 usec\nrounds: 2211"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 928106.2916061399,
            "unit": "iter/sec",
            "range": "stddev: 8.079587520191204e-8",
            "extra": "mean: 1.0774627960655117 usec\nrounds: 176648"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3824.8380497068397,
            "unit": "iter/sec",
            "range": "stddev: 0.000046802764051228024",
            "extra": "mean: 261.4489782323323 usec\nrounds: 781"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 3858.7156160024438,
            "unit": "iter/sec",
            "range": "stddev: 0.000014294758632112725",
            "extra": "mean: 259.15358878817324 usec\nrounds: 3407"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1305.264667448459,
            "unit": "iter/sec",
            "range": "stddev: 0.0029003434133884664",
            "extra": "mean: 766.1281462209556 usec\nrounds: 1402"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 611.8538307907809,
            "unit": "iter/sec",
            "range": "stddev: 0.0029181394773773",
            "extra": "mean: 1.6343772804487726 msec\nrounds: 624"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 615.1443810682784,
            "unit": "iter/sec",
            "range": "stddev: 0.0026222513952406457",
            "extra": "mean: 1.625634616483645 msec\nrounds: 631"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 9591.695265046837,
            "unit": "iter/sec",
            "range": "stddev: 0.000007160977096540576",
            "extra": "mean: 104.25685682948112 usec\nrounds: 2731"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 11841.855137921897,
            "unit": "iter/sec",
            "range": "stddev: 0.00005560728846831777",
            "extra": "mean: 84.44622809120835 usec\nrounds: 8501"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 177571.65122306097,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015976640327573766",
            "extra": "mean: 5.6315295437773765 usec\nrounds: 21798"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1579537.0980075393,
            "unit": "iter/sec",
            "range": "stddev: 8.754718734152081e-8",
            "extra": "mean: 633.0968745599079 nsec\nrounds: 74488"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 277695.45703057706,
            "unit": "iter/sec",
            "range": "stddev: 5.883635710098148e-7",
            "extra": "mean: 3.601067193151417 usec\nrounds: 27354"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 4080.5038670921804,
            "unit": "iter/sec",
            "range": "stddev: 0.00005836210189752099",
            "extra": "mean: 245.0677741208987 usec\nrounds: 1762"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 3149.3563904276857,
            "unit": "iter/sec",
            "range": "stddev: 0.00005313422813496014",
            "extra": "mean: 317.52519436652227 usec\nrounds: 2840"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1762.2122797029979,
            "unit": "iter/sec",
            "range": "stddev: 0.000056287934682255664",
            "extra": "mean: 567.4685232408773 usec\nrounds: 1592"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 9.285835425332657,
            "unit": "iter/sec",
            "range": "stddev: 0.02256874245721857",
            "extra": "mean: 107.69090277778383 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 2.597417277870151,
            "unit": "iter/sec",
            "range": "stddev: 0.08806401552582746",
            "extra": "mean: 384.9978239999956 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.183228335821497,
            "unit": "iter/sec",
            "range": "stddev: 0.022750884390646087",
            "extra": "mean: 108.8941669999915 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 7.724280057787889,
            "unit": "iter/sec",
            "range": "stddev: 0.024866928891155594",
            "extra": "mean: 129.46190357142282 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 1.553243683323174,
            "unit": "iter/sec",
            "range": "stddev: 0.25721204892452093",
            "extra": "mean: 643.8139814999886 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_filesystem",
            "value": 11.045373244571065,
            "unit": "iter/sec",
            "range": "stddev: 0.004934359644624561",
            "extra": "mean: 90.53564581817207 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_filesystem",
            "value": 10.44004296092882,
            "unit": "iter/sec",
            "range": "stddev: 0.018898083037519234",
            "extra": "mean: 95.78504645454379 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 38724.45387587757,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026897453774835906",
            "extra": "mean: 25.823475863733876 usec\nrounds: 9177"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 29809.154677436003,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017867420217123552",
            "extra": "mean: 33.54674128874069 usec\nrounds: 7835"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 39066.704764969196,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012263898059490182",
            "extra": "mean: 25.59724466181985 usec\nrounds: 14424"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 26773.46650821623,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018010078635695742",
            "extra": "mean: 37.35041182258265 usec\nrounds: 10082"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 33731.38996480754,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015116975048938492",
            "extra": "mean: 29.645976671679257 usec\nrounds: 10888"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 26261.923905810407,
            "unit": "iter/sec",
            "range": "stddev: 0.000004159265981602735",
            "extra": "mean: 38.07794141764121 usec\nrounds: 10003"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 34037.58398537998,
            "unit": "iter/sec",
            "range": "stddev: 0.000003560396871342398",
            "extra": "mean: 29.37928850736074 usec\nrounds: 8031"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 24475.875354413576,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022636158390619513",
            "extra": "mean: 40.85655714126181 usec\nrounds: 10369"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 28441.537649180766,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022684616408989545",
            "extra": "mean: 35.15984305541948 usec\nrounds: 8194"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 19329.304287702205,
            "unit": "iter/sec",
            "range": "stddev: 0.000004203512826398912",
            "extra": "mean: 51.73491943195418 usec\nrounds: 7807"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 9754.93082079855,
            "unit": "iter/sec",
            "range": "stddev: 0.000007089706031594943",
            "extra": "mean: 102.51225953011307 usec\nrounds: 3830"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 59785.549236239865,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022424911208120358",
            "extra": "mean: 16.726449999623583 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 30134.28651703181,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019620997223991163",
            "extra": "mean: 33.18479099993965 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 51692.88245315982,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013650015287722432",
            "extra": "mean: 19.345023000141737 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 26412.19398157661,
            "unit": "iter/sec",
            "range": "stddev: 0.000004107973935640122",
            "extra": "mean: 37.861300000201936 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 770.2348884037059,
            "unit": "iter/sec",
            "range": "stddev: 0.000026705226530252174",
            "extra": "mean: 1.2983052508468904 msec\nrounds: 590"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 996.1359918302393,
            "unit": "iter/sec",
            "range": "stddev: 0.0001023573575875951",
            "extra": "mean: 1.0038789966444854 msec\nrounds: 894"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6136.315844979235,
            "unit": "iter/sec",
            "range": "stddev: 0.000006256036419869294",
            "extra": "mean: 162.9642321651688 usec\nrounds: 2425"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 6635.931788358665,
            "unit": "iter/sec",
            "range": "stddev: 0.000006276506219053399",
            "extra": "mean: 150.69473766356188 usec\nrounds: 3911"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 6758.982655983596,
            "unit": "iter/sec",
            "range": "stddev: 0.000010676420081234459",
            "extra": "mean: 147.9512599599171 usec\nrounds: 2008"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4560.337011144627,
            "unit": "iter/sec",
            "range": "stddev: 0.00001705477071205415",
            "extra": "mean: 219.2820393659906 usec\nrounds: 1956"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 2024.0758528990245,
            "unit": "iter/sec",
            "range": "stddev: 0.00002247898125909793",
            "extra": "mean: 494.05263076861934 usec\nrounds: 1495"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 2044.1828666443512,
            "unit": "iter/sec",
            "range": "stddev: 0.00001927937080743548",
            "extra": "mean: 489.1930249085591 usec\nrounds: 1927"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 2478.356475745355,
            "unit": "iter/sec",
            "range": "stddev: 0.000011788181809220548",
            "extra": "mean: 403.49320599622547 usec\nrounds: 2335"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 2464.3391083496476,
            "unit": "iter/sec",
            "range": "stddev: 0.00001614676378384853",
            "extra": "mean: 405.7883091705239 usec\nrounds: 2290"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1787.5394188142109,
            "unit": "iter/sec",
            "range": "stddev: 0.000019985603798139897",
            "extra": "mean: 559.428222658924 usec\nrounds: 1527"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1789.0968411442782,
            "unit": "iter/sec",
            "range": "stddev: 0.000014932238845130809",
            "extra": "mean: 558.9412361604841 usec\nrounds: 1698"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 2153.539435804349,
            "unit": "iter/sec",
            "range": "stddev: 0.000025296079308593428",
            "extra": "mean: 464.35184021902955 usec\nrounds: 2009"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 2156.7539369776296,
            "unit": "iter/sec",
            "range": "stddev: 0.000021352179497901312",
            "extra": "mean: 463.6597540660348 usec\nrounds: 2029"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 597.4934583997962,
            "unit": "iter/sec",
            "range": "stddev: 0.000027340579937403295",
            "extra": "mean: 1.673658491053935 msec\nrounds: 503"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 602.5970187978305,
            "unit": "iter/sec",
            "range": "stddev: 0.00002771436140330994",
            "extra": "mean: 1.6594838155604898 msec\nrounds: 347"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4ad00db3b4eb9b5471b9ba152bbd5d83df120867",
          "message": "[Versioning] Allow any torch version for local builds (#764)",
          "timestamp": "2024-04-29T08:50:28+01:00",
          "tree_id": "cc9457f45621bb4248a219ca4167ada1250e0766",
          "url": "https://github.com/pytorch/tensordict/commit/4ad00db3b4eb9b5471b9ba152bbd5d83df120867"
        },
        "date": 1714377297156,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 58875.00705087817,
            "unit": "iter/sec",
            "range": "stddev: 6.004920933198838e-7",
            "extra": "mean: 16.985135970104043 usec\nrounds: 8090"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 58902.42632449212,
            "unit": "iter/sec",
            "range": "stddev: 8.54606411532972e-7",
            "extra": "mean: 16.977229333321905 usec\nrounds: 17359"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 51414.89974656666,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013230116961175853",
            "extra": "mean: 19.44961489624955 usec\nrounds: 30854"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 51777.552242111495,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013040271809810927",
            "extra": "mean: 19.31338884704334 usec\nrounds: 33283"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 352960.6897866724,
            "unit": "iter/sec",
            "range": "stddev: 3.463011390866334e-7",
            "extra": "mean: 2.8331766934283666 usec\nrounds: 108838"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 3787.8936055315403,
            "unit": "iter/sec",
            "range": "stddev: 0.000021589456657031192",
            "extra": "mean: 263.99896727291366 usec\nrounds: 3300"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 3818.4082657053937,
            "unit": "iter/sec",
            "range": "stddev: 0.000013801564667138228",
            "extra": "mean: 261.8892298608789 usec\nrounds: 3550"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 12837.896869901979,
            "unit": "iter/sec",
            "range": "stddev: 0.000004940686602207394",
            "extra": "mean: 77.89437866138859 usec\nrounds: 9457"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 3779.3188639001396,
            "unit": "iter/sec",
            "range": "stddev: 0.000013336800160497721",
            "extra": "mean: 264.59794370672154 usec\nrounds: 3535"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 13177.60200273917,
            "unit": "iter/sec",
            "range": "stddev: 0.000005201314327894967",
            "extra": "mean: 75.88634106509929 usec\nrounds: 11382"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 3805.9740460485186,
            "unit": "iter/sec",
            "range": "stddev: 0.000012710951867370881",
            "extra": "mean: 262.74482902431544 usec\nrounds: 3673"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 258738.35077844354,
            "unit": "iter/sec",
            "range": "stddev: 2.702589170112918e-7",
            "extra": "mean: 3.864908302118287 usec\nrounds: 115128"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 7241.872429221568,
            "unit": "iter/sec",
            "range": "stddev: 0.00000485976751154991",
            "extra": "mean: 138.08583481323356 usec\nrounds: 5854"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 6973.748987907146,
            "unit": "iter/sec",
            "range": "stddev: 0.000024204398740398915",
            "extra": "mean: 143.39489444401477 usec\nrounds: 6461"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 8488.450318410789,
            "unit": "iter/sec",
            "range": "stddev: 0.000005205877673287457",
            "extra": "mean: 117.80713351541655 usec\nrounds: 7677"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 7344.165036534439,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030604825332822532",
            "extra": "mean: 136.16251745778843 usec\nrounds: 6387"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 8633.084020887538,
            "unit": "iter/sec",
            "range": "stddev: 0.0000052312909087240445",
            "extra": "mean: 115.83346085599585 usec\nrounds: 7766"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 7110.657669161666,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027710318188522835",
            "extra": "mean: 140.63396756349522 usec\nrounds: 6659"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 809398.4606760793,
            "unit": "iter/sec",
            "range": "stddev: 7.65918743465538e-8",
            "extra": "mean: 1.2354854235387525 usec\nrounds: 193462"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 19955.639021575087,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014161266643507022",
            "extra": "mean: 50.111148979937326 usec\nrounds: 15096"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 19901.919809320687,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026462856875575483",
            "extra": "mean: 50.24640886813688 usec\nrounds: 17930"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 21980.343327201957,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013459642356658705",
            "extra": "mean: 45.495194734398964 usec\nrounds: 18158"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 19440.30936814006,
            "unit": "iter/sec",
            "range": "stddev: 0.000002568830570105655",
            "extra": "mean: 51.43951060978791 usec\nrounds: 16259"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 22479.61138564961,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015690062551509574",
            "extra": "mean: 44.48475477820643 usec\nrounds: 18102"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 19516.972852031417,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027766160418258502",
            "extra": "mean: 51.23745406531707 usec\nrounds: 17808"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 737586.3206422816,
            "unit": "iter/sec",
            "range": "stddev: 1.8134711764248494e-7",
            "extra": "mean: 1.3557735169616645 usec\nrounds: 148744"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 284672.3310720139,
            "unit": "iter/sec",
            "range": "stddev: 3.0339689802512935e-7",
            "extra": "mean: 3.5128106628214204 usec\nrounds: 108732"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 281393.3713198962,
            "unit": "iter/sec",
            "range": "stddev: 2.921413853726727e-7",
            "extra": "mean: 3.5537439823455217 usec\nrounds: 116199"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 284900.20327268733,
            "unit": "iter/sec",
            "range": "stddev: 2.8916672794867523e-7",
            "extra": "mean: 3.5100010056604534 usec\nrounds: 53692"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 284230.5571486395,
            "unit": "iter/sec",
            "range": "stddev: 3.191837733632866e-7",
            "extra": "mean: 3.5182705548335744 usec\nrounds: 44697"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 232730.81971222584,
            "unit": "iter/sec",
            "range": "stddev: 4.2978873703892705e-7",
            "extra": "mean: 4.29680951253689 usec\nrounds: 103104"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 222608.96564372425,
            "unit": "iter/sec",
            "range": "stddev: 0.000001829690542878595",
            "extra": "mean: 4.492182051645016 usec\nrounds: 103542"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 74822.03495971329,
            "unit": "iter/sec",
            "range": "stddev: 7.113823089969746e-7",
            "extra": "mean: 13.36504681459725 usec\nrounds: 37424"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 75222.48395930753,
            "unit": "iter/sec",
            "range": "stddev: 5.956231852150498e-7",
            "extra": "mean: 13.293897613657128 usec\nrounds: 41109"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 94591.88034198999,
            "unit": "iter/sec",
            "range": "stddev: 6.380743558890378e-7",
            "extra": "mean: 10.571731911709264 usec\nrounds: 68414"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 98914.34396543242,
            "unit": "iter/sec",
            "range": "stddev: 5.267300318351788e-7",
            "extra": "mean: 10.109757189002536 usec\nrounds: 75672"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 94581.48502159862,
            "unit": "iter/sec",
            "range": "stddev: 7.242507093612998e-7",
            "extra": "mean: 10.572893836162967 usec\nrounds: 50488"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 99469.8252955072,
            "unit": "iter/sec",
            "range": "stddev: 5.10652672835544e-7",
            "extra": "mean: 10.053300053852286 usec\nrounds: 59376"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 91113.92067293446,
            "unit": "iter/sec",
            "range": "stddev: 6.173434153021629e-7",
            "extra": "mean: 10.975271315451707 usec\nrounds: 53119"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 98353.37298249532,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012284945418825289",
            "extra": "mean: 10.167419476076102 usec\nrounds: 70240"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 90502.55661091126,
            "unit": "iter/sec",
            "range": "stddev: 6.165563809730036e-7",
            "extra": "mean: 11.049411612748154 usec\nrounds: 51478"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 98071.72094509473,
            "unit": "iter/sec",
            "range": "stddev: 6.698160217664281e-7",
            "extra": "mean: 10.196619273764435 usec\nrounds: 59200"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2816.23493654641,
            "unit": "iter/sec",
            "range": "stddev: 0.0001480460981114056",
            "extra": "mean: 355.0840119987697 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3370.4642095715294,
            "unit": "iter/sec",
            "range": "stddev: 0.000016457111213159336",
            "extra": "mean: 296.6950360013243 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2253.840336585731,
            "unit": "iter/sec",
            "range": "stddev: 0.002772382668299048",
            "extra": "mean: 443.6871519989154 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3278.519159520164,
            "unit": "iter/sec",
            "range": "stddev: 0.000014909056105873468",
            "extra": "mean: 305.0157559995341 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 10650.894651742225,
            "unit": "iter/sec",
            "range": "stddev: 0.000010257886627918521",
            "extra": "mean: 93.88882649744589 usec\nrounds: 7631"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 2456.085012615382,
            "unit": "iter/sec",
            "range": "stddev: 0.000019626646862378804",
            "extra": "mean: 407.1520305134479 usec\nrounds: 2294"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1418.2958376217282,
            "unit": "iter/sec",
            "range": "stddev: 0.00007578052623725512",
            "extra": "mean: 705.0715185605083 usec\nrounds: 916"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 530759.6102847846,
            "unit": "iter/sec",
            "range": "stddev: 1.8444412648948993e-7",
            "extra": "mean: 1.884092121221205 usec\nrounds: 112146"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 94917.76891666389,
            "unit": "iter/sec",
            "range": "stddev: 5.796094427189429e-7",
            "extra": "mean: 10.535435160491208 usec\nrounds: 20597"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 76006.7269018914,
            "unit": "iter/sec",
            "range": "stddev: 7.185812332617641e-7",
            "extra": "mean: 13.156730210087701 usec\nrounds: 24356"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 59762.64632690666,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015544127330664127",
            "extra": "mean: 16.732860096755363 usec\nrounds: 21908"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 74818.00558626816,
            "unit": "iter/sec",
            "range": "stddev: 0.000001309544874676669",
            "extra": "mean: 13.365766598081258 usec\nrounds: 14233"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 87656.18788437407,
            "unit": "iter/sec",
            "range": "stddev: 8.873740275345619e-7",
            "extra": "mean: 11.408207727663045 usec\nrounds: 10663"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 43184.39897685834,
            "unit": "iter/sec",
            "range": "stddev: 0.000001753280499014987",
            "extra": "mean: 23.156510769916704 usec\nrounds: 9750"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 17336.250431470686,
            "unit": "iter/sec",
            "range": "stddev: 0.000012549616763142936",
            "extra": "mean: 57.68260004970216 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 53425.8005090813,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013694059111348304",
            "extra": "mean: 18.71754827201925 usec\nrounds: 15454"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 24839.738618878706,
            "unit": "iter/sec",
            "range": "stddev: 0.000003873870947001319",
            "extra": "mean: 40.25807257246981 usec\nrounds: 6931"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 29334.401262301224,
            "unit": "iter/sec",
            "range": "stddev: 0.000002715804038107028",
            "extra": "mean: 34.08966799963764 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 16477.98250730155,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034329085477023283",
            "extra": "mean: 60.6870409989142 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 12069.332570251128,
            "unit": "iter/sec",
            "range": "stddev: 0.000005064169046588787",
            "extra": "mean: 82.85462300250401 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 20584.21204772342,
            "unit": "iter/sec",
            "range": "stddev: 0.00000407634370832647",
            "extra": "mean: 48.5809220037936 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 49281.25727381216,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016301767684194061",
            "extra": "mean: 20.291690093129898 usec\nrounds: 17060"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 50514.71318107351,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014521054990820785",
            "extra": "mean: 19.796212569106952 usec\nrounds: 18808"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 7080.904783237093,
            "unit": "iter/sec",
            "range": "stddev: 0.00009191162956268997",
            "extra": "mean: 141.22489012524778 usec\nrounds: 3686"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 46005.847819827075,
            "unit": "iter/sec",
            "range": "stddev: 0.000001989719946636684",
            "extra": "mean: 21.73636716611125 usec\nrounds: 24542"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 33796.24773949618,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025494679004223505",
            "extra": "mean: 29.589083607981255 usec\nrounds: 16099"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 39810.837302689564,
            "unit": "iter/sec",
            "range": "stddev: 0.000002177581001014802",
            "extra": "mean: 25.11878844438274 usec\nrounds: 10645"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 46342.29805632169,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024149781106383987",
            "extra": "mean: 21.578558723709797 usec\nrounds: 17148"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 39085.18918400273,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019464681628258154",
            "extra": "mean: 25.585139048253406 usec\nrounds: 15340"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 25217.349522790057,
            "unit": "iter/sec",
            "range": "stddev: 0.000010105027040154349",
            "extra": "mean: 39.65523811676778 usec\nrounds: 10898"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 16635.029265106445,
            "unit": "iter/sec",
            "range": "stddev: 0.000002348071976020917",
            "extra": "mean: 60.11411125663573 usec\nrounds: 11469"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 8295.27596603564,
            "unit": "iter/sec",
            "range": "stddev: 0.000005838721049373631",
            "extra": "mean: 120.55054034301232 usec\nrounds: 5478"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2532.8817495089743,
            "unit": "iter/sec",
            "range": "stddev: 0.000010441280334823992",
            "extra": "mean: 394.8072191660193 usec\nrounds: 2181"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 930812.4482403079,
            "unit": "iter/sec",
            "range": "stddev: 6.421025585747876e-8",
            "extra": "mean: 1.0743302819923461 usec\nrounds: 176336"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3812.4743224650583,
            "unit": "iter/sec",
            "range": "stddev: 0.00004317256055628195",
            "extra": "mean: 262.2968485603919 usec\nrounds: 799"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 4057.4985905034514,
            "unit": "iter/sec",
            "range": "stddev: 0.000013184151705088522",
            "extra": "mean: 246.45726367976894 usec\nrounds: 3637"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1332.9924410416845,
            "unit": "iter/sec",
            "range": "stddev: 0.003103499761548728",
            "extra": "mean: 750.1918009516519 usec\nrounds: 1472"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 622.945786818325,
            "unit": "iter/sec",
            "range": "stddev: 0.002633092122511587",
            "extra": "mean: 1.6052761270085267 msec\nrounds: 622"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 611.8690384950016,
            "unit": "iter/sec",
            "range": "stddev: 0.003252220807054396",
            "extra": "mean: 1.6343366588047568 msec\nrounds: 636"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 9809.044971846,
            "unit": "iter/sec",
            "range": "stddev: 0.000006706937177576393",
            "extra": "mean: 101.94672395428995 usec\nrounds: 2054"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 12113.569677637897,
            "unit": "iter/sec",
            "range": "stddev: 0.00004346364574723498",
            "extra": "mean: 82.55204919867984 usec\nrounds: 8293"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 181604.71475338502,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022623839054343505",
            "extra": "mean: 5.506464969028897 usec\nrounds: 21952"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1581743.1360018426,
            "unit": "iter/sec",
            "range": "stddev: 9.028177387506586e-8",
            "extra": "mean: 632.2139020167906 nsec\nrounds: 66278"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 273412.1120737132,
            "unit": "iter/sec",
            "range": "stddev: 5.520301541542065e-7",
            "extra": "mean: 3.657482444414881 usec\nrounds: 23953"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 4249.413424450581,
            "unit": "iter/sec",
            "range": "stddev: 0.00005390674580973808",
            "extra": "mean: 235.32659690067527 usec\nrounds: 1806"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 3235.745441500056,
            "unit": "iter/sec",
            "range": "stddev: 0.00004989742213855196",
            "extra": "mean: 309.0477968923325 usec\nrounds: 2767"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1666.683350096985,
            "unit": "iter/sec",
            "range": "stddev: 0.0000554001177200776",
            "extra": "mean: 599.9939940252057 usec\nrounds: 1506"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 7.878801300036228,
            "unit": "iter/sec",
            "range": "stddev: 0.032751219763301476",
            "extra": "mean: 126.92286071428174 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 2.6192825856896484,
            "unit": "iter/sec",
            "range": "stddev: 0.08872315471971982",
            "extra": "mean: 381.783930250009 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 8.263227969518072,
            "unit": "iter/sec",
            "range": "stddev: 0.0355230260700058",
            "extra": "mean: 121.0180820000204 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 7.320185614180025,
            "unit": "iter/sec",
            "range": "stddev: 0.0344486877319966",
            "extra": "mean: 136.60855785717882 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 1.5693810643825892,
            "unit": "iter/sec",
            "range": "stddev: 0.23359330690164454",
            "extra": "mean: 637.1938738749918 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_filesystem",
            "value": 10.022066023735261,
            "unit": "iter/sec",
            "range": "stddev: 0.004907130926289807",
            "extra": "mean: 99.779825600001 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_filesystem",
            "value": 9.395622531169852,
            "unit": "iter/sec",
            "range": "stddev: 0.025569934990633765",
            "extra": "mean: 106.43254309999293 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 38802.91638319434,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014024182831274614",
            "extra": "mean: 25.771258792112416 usec\nrounds: 7933"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 29422.03302293167,
            "unit": "iter/sec",
            "range": "stddev: 0.000001457153049638758",
            "extra": "mean: 33.988133968193 usec\nrounds: 6621"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 38661.338948667646,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014937808426586394",
            "extra": "mean: 25.865632882703412 usec\nrounds: 13527"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 26003.95901737429,
            "unit": "iter/sec",
            "range": "stddev: 0.000003894010738761897",
            "extra": "mean: 38.455682818599264 usec\nrounds: 8812"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 34395.58040259665,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013848583259650528",
            "extra": "mean: 29.073502708635967 usec\nrounds: 9045"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 25978.67976975138,
            "unit": "iter/sec",
            "range": "stddev: 0.000006787720795422417",
            "extra": "mean: 38.493103147003005 usec\nrounds: 9249"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 34501.2269973526,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017532360268080047",
            "extra": "mean: 28.9844764093965 usec\nrounds: 8054"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 24503.80903509367,
            "unit": "iter/sec",
            "range": "stddev: 0.000004688027369631276",
            "extra": "mean: 40.80998176927628 usec\nrounds: 10038"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 28617.243819176903,
            "unit": "iter/sec",
            "range": "stddev: 0.000005062359696539939",
            "extra": "mean: 34.943966173635594 usec\nrounds: 7982"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 18163.94470355332,
            "unit": "iter/sec",
            "range": "stddev: 0.000006436523053670121",
            "extra": "mean: 55.05412047441298 usec\nrounds: 7927"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 10025.118860930957,
            "unit": "iter/sec",
            "range": "stddev: 0.000006777671853638982",
            "extra": "mean: 99.74944076694345 usec\nrounds: 3385"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 56434.91817119611,
            "unit": "iter/sec",
            "range": "stddev: 0.000002517660569889241",
            "extra": "mean: 17.719526002792918 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 28704.486166673792,
            "unit": "iter/sec",
            "range": "stddev: 0.00000212268079327335",
            "extra": "mean: 34.837760000073104 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 49532.762403983186,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015294177000085194",
            "extra": "mean: 20.188658000620308 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 25235.823088085137,
            "unit": "iter/sec",
            "range": "stddev: 0.000002236801787004166",
            "extra": "mean: 39.62620900097136 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 774.876488652581,
            "unit": "iter/sec",
            "range": "stddev: 0.000021425512237666606",
            "extra": "mean: 1.2905282514622198 msec\nrounds: 513"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 990.2333275517113,
            "unit": "iter/sec",
            "range": "stddev: 0.0001168328368808574",
            "extra": "mean: 1.0098630011498766 msec\nrounds: 872"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6210.806998399855,
            "unit": "iter/sec",
            "range": "stddev: 0.000007352437549871802",
            "extra": "mean: 161.00967237552845 usec\nrounds: 2277"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 6809.635574823449,
            "unit": "iter/sec",
            "range": "stddev: 0.000008245254460292679",
            "extra": "mean: 146.85073657938392 usec\nrounds: 3819"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 7051.5249421686685,
            "unit": "iter/sec",
            "range": "stddev: 0.000006944582557372603",
            "extra": "mean: 141.8132968685854 usec\nrounds: 2139"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4551.600003943691,
            "unit": "iter/sec",
            "range": "stddev: 0.000017060168912925252",
            "extra": "mean: 219.70296140556272 usec\nrounds: 2643"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 2067.1611259716537,
            "unit": "iter/sec",
            "range": "stddev: 0.00002310627126688783",
            "extra": "mean: 483.7552271257798 usec\nrounds: 1858"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 2093.8183714723573,
            "unit": "iter/sec",
            "range": "stddev: 0.000024294047671865216",
            "extra": "mean: 477.59634437480247 usec\nrounds: 1992"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 2558.99919281571,
            "unit": "iter/sec",
            "range": "stddev: 0.00001744913811582249",
            "extra": "mean: 390.77777078142924 usec\nrounds: 2430"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 2563.7530833805963,
            "unit": "iter/sec",
            "range": "stddev: 0.000017649842176552197",
            "extra": "mean: 390.05316326383024 usec\nrounds: 2401"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1827.3630945637865,
            "unit": "iter/sec",
            "range": "stddev: 0.000024765080872366893",
            "extra": "mean: 547.2366181493405 usec\nrounds: 1697"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1829.9492541666239,
            "unit": "iter/sec",
            "range": "stddev: 0.00002334768420202807",
            "extra": "mean: 546.4632408374675 usec\nrounds: 1719"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 2210.5895989479914,
            "unit": "iter/sec",
            "range": "stddev: 0.000021398358398752276",
            "extra": "mean: 452.3680019465826 usec\nrounds: 2056"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 2213.573804356242,
            "unit": "iter/sec",
            "range": "stddev: 0.000016326438062258074",
            "extra": "mean: 451.75814695314523 usec\nrounds: 1919"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 583.2375727207113,
            "unit": "iter/sec",
            "range": "stddev: 0.00008362042015919678",
            "extra": "mean: 1.7145671794345445 msec\nrounds: 496"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 596.3325566020604,
            "unit": "iter/sec",
            "range": "stddev: 0.00003626879637765007",
            "extra": "mean: 1.6769166615655893 msec\nrounds: 523"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ad35bfdf958da9fedc0751e5e8b57b9c4fbf623f",
          "message": "[BugFix] Fix map test with fork on cuda (#765)",
          "timestamp": "2024-04-29T10:18:41+01:00",
          "tree_id": "7885be2ffe2c99688becf1071e158136e49a86a0",
          "url": "https://github.com/pytorch/tensordict/commit/ad35bfdf958da9fedc0751e5e8b57b9c4fbf623f"
        },
        "date": 1714382584808,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 64810.202157722466,
            "unit": "iter/sec",
            "range": "stddev: 6.962862123751537e-7",
            "extra": "mean: 15.429669507377781 usec\nrounds: 9035"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 63253.33800398452,
            "unit": "iter/sec",
            "range": "stddev: 9.132726435972551e-7",
            "extra": "mean: 15.80944234021305 usec\nrounds: 13675"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 55218.7087612947,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010318961795590126",
            "extra": "mean: 18.109804130388238 usec\nrounds: 30311"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 56311.377326661364,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010043886043035104",
            "extra": "mean: 17.758400654969183 usec\nrounds: 35724"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 395053.85456066043,
            "unit": "iter/sec",
            "range": "stddev: 2.5193496319125047e-7",
            "extra": "mean: 2.5313004504464347 usec\nrounds: 98136"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 3770.4853392382643,
            "unit": "iter/sec",
            "range": "stddev: 0.000003860142459813992",
            "extra": "mean: 265.2178459874415 usec\nrounds: 2766"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 3802.816967022589,
            "unit": "iter/sec",
            "range": "stddev: 0.000019500809704864335",
            "extra": "mean: 262.9629584257769 usec\nrounds: 3608"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 13085.407126638289,
            "unit": "iter/sec",
            "range": "stddev: 0.000004081635818017911",
            "extra": "mean: 76.42100779304567 usec\nrounds: 5646"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 3746.668957710404,
            "unit": "iter/sec",
            "range": "stddev: 0.000012313086651268174",
            "extra": "mean: 266.90375138216154 usec\nrounds: 3435"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 12859.003967080194,
            "unit": "iter/sec",
            "range": "stddev: 0.000004768201045224128",
            "extra": "mean: 77.7665208409655 usec\nrounds: 10892"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 3773.0089090127326,
            "unit": "iter/sec",
            "range": "stddev: 0.000004027130849232164",
            "extra": "mean: 265.0404555396785 usec\nrounds: 3475"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 254069.05436305207,
            "unit": "iter/sec",
            "range": "stddev: 3.2574733570238305e-7",
            "extra": "mean: 3.9359378201606945 usec\nrounds: 117289"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 7238.320387339813,
            "unit": "iter/sec",
            "range": "stddev: 0.000002871299152783858",
            "extra": "mean: 138.15359731092454 usec\nrounds: 6099"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 7000.192667493227,
            "unit": "iter/sec",
            "range": "stddev: 0.0000225666448088819",
            "extra": "mean: 142.85321097570596 usec\nrounds: 6560"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 8594.56063467903,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024427989970109058",
            "extra": "mean: 116.35266100340286 usec\nrounds: 7593"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 7198.78947454536,
            "unit": "iter/sec",
            "range": "stddev: 0.000005576434775295566",
            "extra": "mean: 138.91224400101726 usec\nrounds: 6459"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 8572.564210468598,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023898617054041614",
            "extra": "mean: 116.65121140519722 usec\nrounds: 7663"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 6998.486420944161,
            "unit": "iter/sec",
            "range": "stddev: 0.00000574099028684743",
            "extra": "mean: 142.88803890614545 usec\nrounds: 6580"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 888170.1834571164,
            "unit": "iter/sec",
            "range": "stddev: 6.455104767357273e-8",
            "extra": "mean: 1.12591034761786 usec\nrounds: 167449"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 19822.333818818697,
            "unit": "iter/sec",
            "range": "stddev: 0.000002521951480381857",
            "extra": "mean: 50.448146476608706 usec\nrounds: 15197"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 19665.771988801556,
            "unit": "iter/sec",
            "range": "stddev: 0.000002221814916080863",
            "extra": "mean: 50.84977088971835 usec\nrounds: 17341"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 21566.1441019655,
            "unit": "iter/sec",
            "range": "stddev: 0.000002349060574223537",
            "extra": "mean: 46.36897515253372 usec\nrounds: 17869"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 19596.96496724215,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027479106353718976",
            "extra": "mean: 51.028309826117344 usec\nrounds: 16222"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 21797.66167593383,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024729563192944794",
            "extra": "mean: 45.876480462309004 usec\nrounds: 17991"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 19841.63921751239,
            "unit": "iter/sec",
            "range": "stddev: 0.000001670005137771211",
            "extra": "mean: 50.399061742710856 usec\nrounds: 17411"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 727785.7709135346,
            "unit": "iter/sec",
            "range": "stddev: 1.5622135077593645e-7",
            "extra": "mean: 1.374030710637246 usec\nrounds: 138812"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 285829.51530923817,
            "unit": "iter/sec",
            "range": "stddev: 3.301191655060972e-7",
            "extra": "mean: 3.498589006520558 usec\nrounds: 101133"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 285639.67723580933,
            "unit": "iter/sec",
            "range": "stddev: 3.595663713107471e-7",
            "extra": "mean: 3.500914192584148 usec\nrounds: 116878"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 290510.8541176163,
            "unit": "iter/sec",
            "range": "stddev: 3.1113118367063226e-7",
            "extra": "mean: 3.4422121783964044 usec\nrounds: 60680"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 289076.89044504636,
            "unit": "iter/sec",
            "range": "stddev: 2.6920744118139374e-7",
            "extra": "mean: 3.4592872452047505 usec\nrounds: 48335"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 237896.0655471985,
            "unit": "iter/sec",
            "range": "stddev: 5.029667972773511e-7",
            "extra": "mean: 4.203516345257086 usec\nrounds: 94340"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 231767.1232222373,
            "unit": "iter/sec",
            "range": "stddev: 8.0413731703482e-7",
            "extra": "mean: 4.314675809481045 usec\nrounds: 97381"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 237391.60219794067,
            "unit": "iter/sec",
            "range": "stddev: 4.952957173681767e-7",
            "extra": "mean: 4.21244892717892 usec\nrounds: 68461"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 237526.3979603911,
            "unit": "iter/sec",
            "range": "stddev: 3.827917079051977e-7",
            "extra": "mean: 4.210058370719518 usec\nrounds: 72913"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 92475.16876907981,
            "unit": "iter/sec",
            "range": "stddev: 5.160709161953126e-7",
            "extra": "mean: 10.813713706185329 usec\nrounds: 67532"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 96923.64812604462,
            "unit": "iter/sec",
            "range": "stddev: 6.697561738837407e-7",
            "extra": "mean: 10.317399513269944 usec\nrounds: 72701"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 92663.92850895654,
            "unit": "iter/sec",
            "range": "stddev: 5.706892283739402e-7",
            "extra": "mean: 10.791685784219087 usec\nrounds: 58062"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 98816.46216550657,
            "unit": "iter/sec",
            "range": "stddev: 6.400972015972131e-7",
            "extra": "mean: 10.119771322364398 usec\nrounds: 60640"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 88113.08143252265,
            "unit": "iter/sec",
            "range": "stddev: 7.855462707599147e-7",
            "extra": "mean: 11.349052646238505 usec\nrounds: 46727"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 96258.21010524388,
            "unit": "iter/sec",
            "range": "stddev: 4.914713418290755e-7",
            "extra": "mean: 10.38872423356564 usec\nrounds: 66016"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 89165.43263131478,
            "unit": "iter/sec",
            "range": "stddev: 7.808544756940093e-7",
            "extra": "mean: 11.215108484190782 usec\nrounds: 52450"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 93896.98910659291,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023727099215884825",
            "extra": "mean: 10.649968753149144 usec\nrounds: 59270"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2821.03215444394,
            "unit": "iter/sec",
            "range": "stddev: 0.00015987970478753961",
            "extra": "mean: 354.48018500062517 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3202.115250726943,
            "unit": "iter/sec",
            "range": "stddev: 0.000014724113152984627",
            "extra": "mean: 312.2935690003601 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2432.9829967599503,
            "unit": "iter/sec",
            "range": "stddev: 0.0017372077596285772",
            "extra": "mean: 411.0180799996215 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3078.0512347874733,
            "unit": "iter/sec",
            "range": "stddev: 0.000013273300227788498",
            "extra": "mean: 324.88088200034326 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 10523.857001094877,
            "unit": "iter/sec",
            "range": "stddev: 0.0000072733148121959935",
            "extra": "mean: 95.02219574971063 usec\nrounds: 7576"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 2424.4978914242524,
            "unit": "iter/sec",
            "range": "stddev: 0.00001862161012754021",
            "extra": "mean: 412.45653524266737 usec\nrounds: 2270"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1467.3530084458077,
            "unit": "iter/sec",
            "range": "stddev: 0.00009237661141116326",
            "extra": "mean: 681.499267213948 usec\nrounds: 973"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 526651.5618093996,
            "unit": "iter/sec",
            "range": "stddev: 2.2964730948590866e-7",
            "extra": "mean: 1.8987886346796972 usec\nrounds: 93284"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 121908.14754590066,
            "unit": "iter/sec",
            "range": "stddev: 5.222118184335864e-7",
            "extra": "mean: 8.202897182269803 usec\nrounds: 23245"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 91663.97921064787,
            "unit": "iter/sec",
            "range": "stddev: 8.045039164130709e-7",
            "extra": "mean: 10.909410747944467 usec\nrounds: 24823"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 69403.95675636854,
            "unit": "iter/sec",
            "range": "stddev: 0.000001276612441860038",
            "extra": "mean: 14.40840042463774 usec\nrounds: 23977"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 72683.24570534629,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030130960033629244",
            "extra": "mean: 13.758328900912637 usec\nrounds: 13764"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 85770.39227360867,
            "unit": "iter/sec",
            "range": "stddev: 8.103610222057812e-7",
            "extra": "mean: 11.659034936088283 usec\nrounds: 11020"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 42918.194748921815,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022889413748142576",
            "extra": "mean: 23.300141253614168 usec\nrounds: 10329"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 16610.90937667771,
            "unit": "iter/sec",
            "range": "stddev: 0.000013213135814663085",
            "extra": "mean: 60.20140001510299 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 50236.111237630554,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022728783940073714",
            "extra": "mean: 19.905999396922393 usec\nrounds: 14911"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 23519.447134273967,
            "unit": "iter/sec",
            "range": "stddev: 0.000004765800017037082",
            "extra": "mean: 42.51800623930225 usec\nrounds: 7213"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 28946.316452996765,
            "unit": "iter/sec",
            "range": "stddev: 0.000002864465965496337",
            "extra": "mean: 34.54670999758491 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 16798.517107605938,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031307317185286214",
            "extra": "mean: 59.529063999775644 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 11906.607996191344,
            "unit": "iter/sec",
            "range": "stddev: 0.000004493911265688351",
            "extra": "mean: 83.98697599852767 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 20318.42218375791,
            "unit": "iter/sec",
            "range": "stddev: 0.000005451820983016788",
            "extra": "mean: 49.216420003290295 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 52300.04258818473,
            "unit": "iter/sec",
            "range": "stddev: 0.000001664730705083803",
            "extra": "mean: 19.120443321128633 usec\nrounds: 18305"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 53198.34008190265,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018056520032968348",
            "extra": "mean: 18.797578993262356 usec\nrounds: 16128"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 6918.55369065637,
            "unit": "iter/sec",
            "range": "stddev: 0.00010176882038904261",
            "extra": "mean: 144.5388797590048 usec\nrounds: 3493"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 51785.29586586484,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019234064840174378",
            "extra": "mean: 19.310500853180738 usec\nrounds: 27527"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 35612.16467405084,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028776329300914427",
            "extra": "mean: 28.080292482996974 usec\nrounds: 15471"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 39307.02438166124,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023677606651705907",
            "extra": "mean: 25.44074540698511 usec\nrounds: 11320"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 48092.81057719845,
            "unit": "iter/sec",
            "range": "stddev: 0.00000208182877756263",
            "extra": "mean: 20.793128702569852 usec\nrounds: 17350"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 40027.27877972845,
            "unit": "iter/sec",
            "range": "stddev: 0.000002373117889092371",
            "extra": "mean: 24.982962381805567 usec\nrounds: 15418"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 25403.69974037782,
            "unit": "iter/sec",
            "range": "stddev: 0.0000038319416277281815",
            "extra": "mean: 39.364344966278814 usec\nrounds: 10946"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 16206.715178256582,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031086855317581048",
            "extra": "mean: 61.70281818376312 usec\nrounds: 990"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 8065.764141691111,
            "unit": "iter/sec",
            "range": "stddev: 0.000008442030663931526",
            "extra": "mean: 123.98081352653273 usec\nrounds: 5175"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2504.8729083229127,
            "unit": "iter/sec",
            "range": "stddev: 0.000008449969632605898",
            "extra": "mean: 399.221851407036 usec\nrounds: 2167"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 920687.191800055,
            "unit": "iter/sec",
            "range": "stddev: 8.498610289668445e-8",
            "extra": "mean: 1.0861452281581914 usec\nrounds: 175101"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3784.432386357413,
            "unit": "iter/sec",
            "range": "stddev: 0.000046149037107589965",
            "extra": "mean: 264.2404191457939 usec\nrounds: 773"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 3866.360510805986,
            "unit": "iter/sec",
            "range": "stddev: 0.000008011206367661453",
            "extra": "mean: 258.64116840763484 usec\nrounds: 3444"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1269.7649370389092,
            "unit": "iter/sec",
            "range": "stddev: 0.0031047950764727973",
            "extra": "mean: 787.5473411102366 usec\nrounds: 1457"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 617.5709620084875,
            "unit": "iter/sec",
            "range": "stddev: 0.002788937776774425",
            "extra": "mean: 1.6192471173640715 msec\nrounds: 622"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 614.3445221700487,
            "unit": "iter/sec",
            "range": "stddev: 0.002738923048119431",
            "extra": "mean: 1.6277511459981784 msec\nrounds: 637"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 9006.711950848143,
            "unit": "iter/sec",
            "range": "stddev: 0.00009324087824546266",
            "extra": "mean: 111.02830927171289 usec\nrounds: 2535"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 12033.656858538723,
            "unit": "iter/sec",
            "range": "stddev: 0.000008525892202246811",
            "extra": "mean: 83.10025886191278 usec\nrounds: 8491"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 168973.50820202945,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018458313698890497",
            "extra": "mean: 5.918087460221114 usec\nrounds: 22696"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1586682.122891272,
            "unit": "iter/sec",
            "range": "stddev: 9.387140546831055e-8",
            "extra": "mean: 630.2459614139898 nsec\nrounds: 68649"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 270539.7413898805,
            "unit": "iter/sec",
            "range": "stddev: 4.5040629118444027e-7",
            "extra": "mean: 3.6963146148605173 usec\nrounds: 25180"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 4054.050626553876,
            "unit": "iter/sec",
            "range": "stddev: 0.00005448615248261909",
            "extra": "mean: 246.66687521118715 usec\nrounds: 1779"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 3103.5533129536393,
            "unit": "iter/sec",
            "range": "stddev: 0.00006123133536553837",
            "extra": "mean: 322.2113168883521 usec\nrounds: 2777"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1778.086223508589,
            "unit": "iter/sec",
            "range": "stddev: 0.00005924525441982028",
            "extra": "mean: 562.402422772705 usec\nrounds: 1651"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 8.986826121718696,
            "unit": "iter/sec",
            "range": "stddev: 0.023102348568556096",
            "extra": "mean: 111.27399000001503 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 2.5764206998457295,
            "unit": "iter/sec",
            "range": "stddev: 0.08591310670288686",
            "extra": "mean: 388.13536937499293 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.088067861401333,
            "unit": "iter/sec",
            "range": "stddev: 0.024028007317760103",
            "extra": "mean: 110.03438962501377 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 7.502206294374972,
            "unit": "iter/sec",
            "range": "stddev: 0.029155586447205183",
            "extra": "mean: 133.29412185716396 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 2.4016650861725517,
            "unit": "iter/sec",
            "range": "stddev: 0.06315921583261701",
            "extra": "mean: 416.37778962497407 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_filesystem",
            "value": 10.642530032914236,
            "unit": "iter/sec",
            "range": "stddev: 0.003955464257299636",
            "extra": "mean: 93.96261949999598 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_filesystem",
            "value": 9.818188998605208,
            "unit": "iter/sec",
            "range": "stddev: 0.02053617829129389",
            "extra": "mean: 101.85177736363212 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 38080.470526805504,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014769772884413209",
            "extra": "mean: 26.26017972377948 usec\nrounds: 9331"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 29314.21507227599,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021707137158554217",
            "extra": "mean: 34.11314263521772 usec\nrounds: 7740"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 38726.89456580447,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015670445887326428",
            "extra": "mean: 25.821848387580033 usec\nrounds: 13330"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 26188.0194597176,
            "unit": "iter/sec",
            "range": "stddev: 0.000002322091970317171",
            "extra": "mean: 38.18540006578961 usec\nrounds: 9171"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 33530.14950887224,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037170913621121194",
            "extra": "mean: 29.82390519121888 usec\nrounds: 10231"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 25633.031947219293,
            "unit": "iter/sec",
            "range": "stddev: 0.000004907249530581577",
            "extra": "mean: 39.01216219989463 usec\nrounds: 10672"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 33555.20964058992,
            "unit": "iter/sec",
            "range": "stddev: 0.000001933753532004012",
            "extra": "mean: 29.801631720111622 usec\nrounds: 8241"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 24384.083013056672,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025109469049335094",
            "extra": "mean: 41.01035907171662 usec\nrounds: 9778"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 27252.01392318964,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027882571235428254",
            "extra": "mean: 36.69453578067737 usec\nrounds: 8175"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 19490.434934979094,
            "unit": "iter/sec",
            "range": "stddev: 0.0000066232137987296",
            "extra": "mean: 51.30721830149208 usec\nrounds: 7792"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 9721.781902679364,
            "unit": "iter/sec",
            "range": "stddev: 0.000010235399367466644",
            "extra": "mean: 102.8618014691726 usec\nrounds: 3813"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 62704.50682380345,
            "unit": "iter/sec",
            "range": "stddev: 0.000001180814396814462",
            "extra": "mean: 15.947817001574549 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 31824.185883380524,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027002276351016588",
            "extra": "mean: 31.42264200141653 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 52734.43329618428,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018503787384552525",
            "extra": "mean: 18.962942000030125 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 27471.019585778107,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028945197145885305",
            "extra": "mean: 36.40199799929178 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 763.1101108222161,
            "unit": "iter/sec",
            "range": "stddev: 0.000024520750805164777",
            "extra": "mean: 1.3104268778755215 msec\nrounds: 565"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 965.934208496173,
            "unit": "iter/sec",
            "range": "stddev: 0.00011537385076871228",
            "extra": "mean: 1.0352671964655469 msec\nrounds: 906"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6103.820919528005,
            "unit": "iter/sec",
            "range": "stddev: 0.000008820934792872309",
            "extra": "mean: 163.8318052223144 usec\nrounds: 2259"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 6454.95539475634,
            "unit": "iter/sec",
            "range": "stddev: 0.000012456742107933341",
            "extra": "mean: 154.91973822349672 usec\nrounds: 3736"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 6652.401991513446,
            "unit": "iter/sec",
            "range": "stddev: 0.000009315264707927955",
            "extra": "mean: 150.32164341176505 usec\nrounds: 2064"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4477.943512610089,
            "unit": "iter/sec",
            "range": "stddev: 0.000020297837069392634",
            "extra": "mean: 223.31679646783292 usec\nrounds: 2491"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 2048.1436381661915,
            "unit": "iter/sec",
            "range": "stddev: 0.00002629970814672986",
            "extra": "mean: 488.2470063942153 usec\nrounds: 1876"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 2044.9622331555222,
            "unit": "iter/sec",
            "range": "stddev: 0.00003353316021950521",
            "extra": "mean: 489.0065859343176 usec\nrounds: 1891"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 2507.12643195574,
            "unit": "iter/sec",
            "range": "stddev: 0.000019309701331576087",
            "extra": "mean: 398.8630119542585 usec\nrounds: 2342"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 2500.8613439759824,
            "unit": "iter/sec",
            "range": "stddev: 0.000021355815679067846",
            "extra": "mean: 399.8622324299494 usec\nrounds: 2362"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1793.646267954647,
            "unit": "iter/sec",
            "range": "stddev: 0.000029375727991905862",
            "extra": "mean: 557.5235306236453 usec\nrounds: 1649"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1809.8863404347965,
            "unit": "iter/sec",
            "range": "stddev: 0.00002600682606139809",
            "extra": "mean: 552.5208835819855 usec\nrounds: 1675"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 2168.1613168882195,
            "unit": "iter/sec",
            "range": "stddev: 0.00002397658225227975",
            "extra": "mean: 461.2202939932608 usec\nrounds: 1881"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 2168.9400207504955,
            "unit": "iter/sec",
            "range": "stddev: 0.00002497506302271921",
            "extra": "mean: 461.0547043407778 usec\nrounds: 1867"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 595.7947955694755,
            "unit": "iter/sec",
            "range": "stddev: 0.000017978509059171562",
            "extra": "mean: 1.6784302371157425 msec\nrounds: 485"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 606.7325385719504,
            "unit": "iter/sec",
            "range": "stddev: 0.000018475164214956058",
            "extra": "mean: 1.648172689656092 msec\nrounds: 319"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0c72dd724d513b353bb9dfc669dff947358a7bdf",
          "message": "[BugFix] Sync cuda only if initialized (#767)",
          "timestamp": "2024-04-30T10:05:48+01:00",
          "tree_id": "4a674e627c880dec3389b663ec1d03c1ffe08540",
          "url": "https://github.com/pytorch/tensordict/commit/0c72dd724d513b353bb9dfc669dff947358a7bdf"
        },
        "date": 1714468221159,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 60862.65642149562,
            "unit": "iter/sec",
            "range": "stddev: 7.791351347166882e-7",
            "extra": "mean: 16.43043631015122 usec\nrounds: 8141"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 60050.3848330169,
            "unit": "iter/sec",
            "range": "stddev: 7.972104861921761e-7",
            "extra": "mean: 16.652682622779466 usec\nrounds: 16838"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 52843.77681199935,
            "unit": "iter/sec",
            "range": "stddev: 9.921578720567943e-7",
            "extra": "mean: 18.92370417727084 usec\nrounds: 30951"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 52703.55152254418,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010295678652879921",
            "extra": "mean: 18.97405338181518 usec\nrounds: 33813"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 392713.5428300221,
            "unit": "iter/sec",
            "range": "stddev: 1.9440276401571856e-7",
            "extra": "mean: 2.546385318911269 usec\nrounds: 115795"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 3788.972799759644,
            "unit": "iter/sec",
            "range": "stddev: 0.000010896460789673206",
            "extra": "mean: 263.9237737635477 usec\nrounds: 2325"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 3731.0037580608932,
            "unit": "iter/sec",
            "range": "stddev: 0.000015925378828245822",
            "extra": "mean: 268.02438830019514 usec\nrounds: 3590"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 12856.275200418184,
            "unit": "iter/sec",
            "range": "stddev: 0.000005431492276331761",
            "extra": "mean: 77.78302691960673 usec\nrounds: 9547"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 3762.345563226525,
            "unit": "iter/sec",
            "range": "stddev: 0.0000204647160095789",
            "extra": "mean: 265.79164066535577 usec\nrounds: 3487"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 12930.495037670213,
            "unit": "iter/sec",
            "range": "stddev: 0.000007161651348653513",
            "extra": "mean: 77.33655958930538 usec\nrounds: 10908"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 3728.305438672079,
            "unit": "iter/sec",
            "range": "stddev: 0.000012055413061690304",
            "extra": "mean: 268.2183679554357 usec\nrounds: 3389"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 256489.8079671597,
            "unit": "iter/sec",
            "range": "stddev: 2.655973930796582e-7",
            "extra": "mean: 3.8987903961004076 usec\nrounds: 117703"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 7140.684265604759,
            "unit": "iter/sec",
            "range": "stddev: 0.000005126644130928744",
            "extra": "mean: 140.04260135359843 usec\nrounds: 6058"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 6977.493413080661,
            "unit": "iter/sec",
            "range": "stddev: 0.000023702926941238906",
            "extra": "mean: 143.31794253295985 usec\nrounds: 6421"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 8445.947476415444,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027926364606170998",
            "extra": "mean: 118.3999785450254 usec\nrounds: 7504"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 7109.671829958448,
            "unit": "iter/sec",
            "range": "stddev: 0.000003217341534309529",
            "extra": "mean: 140.65346810892737 usec\nrounds: 6240"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 8468.839335547047,
            "unit": "iter/sec",
            "range": "stddev: 0.000002966630142513139",
            "extra": "mean: 118.0799352046516 usec\nrounds: 7624"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 6974.07078687364,
            "unit": "iter/sec",
            "range": "stddev: 0.0000043672459892874366",
            "extra": "mean: 143.38827788816914 usec\nrounds: 6535"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 860261.1736193792,
            "unit": "iter/sec",
            "range": "stddev: 6.590152157466601e-8",
            "extra": "mean: 1.1624376766798592 usec\nrounds: 194213"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 19873.47369137065,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017589750285558061",
            "extra": "mean: 50.31832962519353 usec\nrounds: 15281"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 19833.768603901717,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015200098065504074",
            "extra": "mean: 50.4190615495675 usec\nrounds: 17953"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 22148.84778192818,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016063069450098782",
            "extra": "mean: 45.14907546639631 usec\nrounds: 17796"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 19735.124877467053,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016900392087798065",
            "extra": "mean: 50.67107536480647 usec\nrounds: 11504"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 22203.660967046388,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023551412086268837",
            "extra": "mean: 45.037617962378015 usec\nrounds: 18205"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 19807.003600721928,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014894498685774382",
            "extra": "mean: 50.487192316335616 usec\nrounds: 17570"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 733148.2119027012,
            "unit": "iter/sec",
            "range": "stddev: 1.6780932853948507e-7",
            "extra": "mean: 1.3639806846213978 usec\nrounds: 166086"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 289652.02189233905,
            "unit": "iter/sec",
            "range": "stddev: 2.707558376650365e-7",
            "extra": "mean: 3.452418503647424 usec\nrounds: 109686"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 288627.31453864154,
            "unit": "iter/sec",
            "range": "stddev: 3.386197402779156e-7",
            "extra": "mean: 3.4646755508862954 usec\nrounds: 123702"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 289856.22865793866,
            "unit": "iter/sec",
            "range": "stddev: 2.853761326778315e-7",
            "extra": "mean: 3.449986238453778 usec\nrounds: 73175"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 290275.20582459355,
            "unit": "iter/sec",
            "range": "stddev: 3.515959341458951e-7",
            "extra": "mean: 3.445006600406224 usec\nrounds: 53178"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 220604.72571733035,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020021238007418325",
            "extra": "mean: 4.532994462146473 usec\nrounds: 104734"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 238292.0352348453,
            "unit": "iter/sec",
            "range": "stddev: 3.686569710622951e-7",
            "extra": "mean: 4.196531365450231 usec\nrounds: 107794"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 239502.36055014047,
            "unit": "iter/sec",
            "range": "stddev: 4.850421901383365e-7",
            "extra": "mean: 4.175324191807484 usec\nrounds: 74823"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 237054.19663676244,
            "unit": "iter/sec",
            "range": "stddev: 3.385150639544895e-7",
            "extra": "mean: 4.218444618098441 usec\nrounds: 79981"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 95638.37636568616,
            "unit": "iter/sec",
            "range": "stddev: 7.043983833492937e-7",
            "extra": "mean: 10.456053709824243 usec\nrounds: 70043"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 100835.88943979675,
            "unit": "iter/sec",
            "range": "stddev: 5.412173535374228e-7",
            "extra": "mean: 9.917103975138156 usec\nrounds: 76249"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 95920.65032779552,
            "unit": "iter/sec",
            "range": "stddev: 5.559097565278247e-7",
            "extra": "mean: 10.425283779693306 usec\nrounds: 59592"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 99538.19845298745,
            "unit": "iter/sec",
            "range": "stddev: 6.65114918662171e-7",
            "extra": "mean: 10.046394404780257 usec\nrounds: 63412"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 90818.00470502705,
            "unit": "iter/sec",
            "range": "stddev: 0.00000133294743063153",
            "extra": "mean: 11.011032484670377 usec\nrounds: 52671"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 100603.73921629543,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011851847081914226",
            "extra": "mean: 9.93998839198239 usec\nrounds: 66592"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 90788.23523980714,
            "unit": "iter/sec",
            "range": "stddev: 7.769847586692412e-7",
            "extra": "mean: 11.014643002571974 usec\nrounds: 54964"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 95800.88008764993,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015402478700680107",
            "extra": "mean: 10.438317467282996 usec\nrounds: 60570"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2845.991174105988,
            "unit": "iter/sec",
            "range": "stddev: 0.00014600435804933517",
            "extra": "mean: 351.3714340010665 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3199.0395920115648,
            "unit": "iter/sec",
            "range": "stddev: 0.000012270750375352484",
            "extra": "mean: 312.5938179999821 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2494.7152451174557,
            "unit": "iter/sec",
            "range": "stddev: 0.001570141158114643",
            "extra": "mean: 400.84735200025534 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3087.050149696849,
            "unit": "iter/sec",
            "range": "stddev: 0.000016568923148746693",
            "extra": "mean: 323.9338369991174 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 10398.665152848585,
            "unit": "iter/sec",
            "range": "stddev: 0.000007714502814619516",
            "extra": "mean: 96.16618915035095 usec\nrounds: 6857"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 2454.5306838095817,
            "unit": "iter/sec",
            "range": "stddev: 0.0000072403341121391625",
            "extra": "mean: 407.4098590806528 usec\nrounds: 2285"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1434.434780238243,
            "unit": "iter/sec",
            "range": "stddev: 0.00016000426714059921",
            "extra": "mean: 697.1387014430251 usec\nrounds: 901"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 529808.6764434297,
            "unit": "iter/sec",
            "range": "stddev: 1.8414698898210315e-7",
            "extra": "mean: 1.8874738079280493 usec\nrounds: 116199"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 100094.53157148145,
            "unit": "iter/sec",
            "range": "stddev: 4.664955886068928e-7",
            "extra": "mean: 9.990555770630293 usec\nrounds: 20683"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 78264.98983046919,
            "unit": "iter/sec",
            "range": "stddev: 7.996890875789788e-7",
            "extra": "mean: 12.777105090872855 usec\nrounds: 23256"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 62489.56908543298,
            "unit": "iter/sec",
            "range": "stddev: 8.846997444289558e-7",
            "extra": "mean: 16.002670759864642 usec\nrounds: 23536"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 72389.60607230925,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013123913796759104",
            "extra": "mean: 13.81413788881666 usec\nrounds: 14664"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 86311.12746562084,
            "unit": "iter/sec",
            "range": "stddev: 6.040698569925312e-7",
            "extra": "mean: 11.585991625451964 usec\nrounds: 12060"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 43119.832511039465,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026036178900256755",
            "extra": "mean: 23.191184700079294 usec\nrounds: 10471"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 16579.733596755144,
            "unit": "iter/sec",
            "range": "stddev: 0.000012458257368925869",
            "extra": "mean: 60.31460000031075 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 51610.816651396846,
            "unit": "iter/sec",
            "range": "stddev: 0.000001375388564993705",
            "extra": "mean: 19.375783312138985 usec\nrounds: 15640"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 24314.311558647358,
            "unit": "iter/sec",
            "range": "stddev: 0.000003595089755165578",
            "extra": "mean: 41.1280408901543 usec\nrounds: 8584"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 27616.41964003138,
            "unit": "iter/sec",
            "range": "stddev: 0.000004392598098411152",
            "extra": "mean: 36.21034200068607 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 16202.930163880945,
            "unit": "iter/sec",
            "range": "stddev: 0.000004219434936281539",
            "extra": "mean: 61.717231999750766 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 11908.287750251657,
            "unit": "iter/sec",
            "range": "stddev: 0.0000047393784848402865",
            "extra": "mean: 83.97512900029369 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 19062.909679951776,
            "unit": "iter/sec",
            "range": "stddev: 0.0000050209587902809774",
            "extra": "mean: 52.4578889995837 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 49083.56517197539,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016951066267743514",
            "extra": "mean: 20.37341820008945 usec\nrounds: 18912"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 50945.96909641489,
            "unit": "iter/sec",
            "range": "stddev: 0.000001790827933851612",
            "extra": "mean: 19.628638295357717 usec\nrounds: 18070"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 7098.637092479357,
            "unit": "iter/sec",
            "range": "stddev: 0.00007129787760050016",
            "extra": "mean: 140.87211206492705 usec\nrounds: 3846"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 46638.50774696422,
            "unit": "iter/sec",
            "range": "stddev: 0.000002150754800825236",
            "extra": "mean: 21.44150935157422 usec\nrounds: 24221"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 33410.46551483056,
            "unit": "iter/sec",
            "range": "stddev: 0.000002073268334429855",
            "extra": "mean: 29.93074129889361 usec\nrounds: 16981"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 38039.52104928242,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020581574373633955",
            "extra": "mean: 26.288448761077767 usec\nrounds: 11661"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 46121.085971459106,
            "unit": "iter/sec",
            "range": "stddev: 0.000001732488375668432",
            "extra": "mean: 21.68205667617682 usec\nrounds: 16409"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 38491.89483651117,
            "unit": "iter/sec",
            "range": "stddev: 0.000002929201481090269",
            "extra": "mean: 25.97949527419622 usec\nrounds: 14812"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 23993.87010434748,
            "unit": "iter/sec",
            "range": "stddev: 0.000007143553426160102",
            "extra": "mean: 41.67731156545724 usec\nrounds: 10367"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 16445.374792646595,
            "unit": "iter/sec",
            "range": "stddev: 0.000008911365340547428",
            "extra": "mean: 60.80737061992294 usec\nrounds: 11273"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 8284.452461514462,
            "unit": "iter/sec",
            "range": "stddev: 0.0000041201180919968115",
            "extra": "mean: 120.70803769416433 usec\nrounds: 5014"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2549.415938679972,
            "unit": "iter/sec",
            "range": "stddev: 0.000007816557115092777",
            "extra": "mean: 392.24670436389306 usec\nrounds: 2131"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 910770.3327481919,
            "unit": "iter/sec",
            "range": "stddev: 1.0878825445642086e-7",
            "extra": "mean: 1.0979716444897472 usec\nrounds: 174795"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3803.3076855928516,
            "unit": "iter/sec",
            "range": "stddev: 0.000008430137867005857",
            "extra": "mean: 262.9290298515835 usec\nrounds: 804"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 3866.116554941115,
            "unit": "iter/sec",
            "range": "stddev: 0.000006587737872738549",
            "extra": "mean: 258.6574889269553 usec\nrounds: 3477"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1284.0094181483373,
            "unit": "iter/sec",
            "range": "stddev: 0.0029127038506614644",
            "extra": "mean: 778.8104867969693 usec\nrounds: 1477"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 618.140791528607,
            "unit": "iter/sec",
            "range": "stddev: 0.0024123606015819276",
            "extra": "mean: 1.617754423756907 msec\nrounds: 623"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 617.2920828128251,
            "unit": "iter/sec",
            "range": "stddev: 0.0025129616417304598",
            "extra": "mean: 1.619978658147183 msec\nrounds: 626"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 9694.387698318684,
            "unit": "iter/sec",
            "range": "stddev: 0.0000072443583217576874",
            "extra": "mean: 103.15246626390152 usec\nrounds: 2816"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 11872.827473894453,
            "unit": "iter/sec",
            "range": "stddev: 0.00004897910459418698",
            "extra": "mean: 84.22593541418539 usec\nrounds: 8841"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 173692.44571418082,
            "unit": "iter/sec",
            "range": "stddev: 0.000002187002631017247",
            "extra": "mean: 5.757302776688098 usec\nrounds: 23192"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1474429.5219158777,
            "unit": "iter/sec",
            "range": "stddev: 2.488924372126622e-7",
            "extra": "mean: 678.2284165747017 nsec\nrounds: 80756"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 268193.0444205459,
            "unit": "iter/sec",
            "range": "stddev: 4.1996552179960834e-7",
            "extra": "mean: 3.7286574756649102 usec\nrounds: 27309"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 4133.88684567095,
            "unit": "iter/sec",
            "range": "stddev: 0.000054669377999112555",
            "extra": "mean: 241.90308959404888 usec\nrounds: 1797"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 3139.846838094765,
            "unit": "iter/sec",
            "range": "stddev: 0.00005574520610142348",
            "extra": "mean: 318.48687262936437 usec\nrounds: 2795"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1639.1279026782834,
            "unit": "iter/sec",
            "range": "stddev: 0.00006646330067094093",
            "extra": "mean: 610.0805180401306 usec\nrounds: 1469"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 8.787808807294851,
            "unit": "iter/sec",
            "range": "stddev: 0.02514244007836277",
            "extra": "mean: 113.79400962500341 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 2.6417740164775374,
            "unit": "iter/sec",
            "range": "stddev: 0.12382883297448458",
            "extra": "mean: 378.53351337498964 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.257848755282172,
            "unit": "iter/sec",
            "range": "stddev: 0.022886645662455317",
            "extra": "mean: 108.01645462499465 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 7.601696456263327,
            "unit": "iter/sec",
            "range": "stddev: 0.02401463861256956",
            "extra": "mean: 131.54958314286043 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 2.3979301215861843,
            "unit": "iter/sec",
            "range": "stddev: 0.048263599864343636",
            "extra": "mean: 417.02633075000506 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_filesystem",
            "value": 10.810445534620504,
            "unit": "iter/sec",
            "range": "stddev: 0.004853528990327275",
            "extra": "mean: 92.50312550000785 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_filesystem",
            "value": 10.257976039612425,
            "unit": "iter/sec",
            "range": "stddev: 0.021215797110253082",
            "extra": "mean: 97.48511754544738 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 38063.54097907886,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016422417031385307",
            "extra": "mean: 26.271859482270376 usec\nrounds: 9038"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 30047.467495191264,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017437266175091553",
            "extra": "mean: 33.280674990663954 usec\nrounds: 8209"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 39119.04970820655,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014650521943127408",
            "extra": "mean: 25.562993157019765 usec\nrounds: 14759"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 27258.814430283295,
            "unit": "iter/sec",
            "range": "stddev: 0.000002406089886109551",
            "extra": "mean: 36.685381257412494 usec\nrounds: 9668"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 33331.66657594331,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016475228310149485",
            "extra": "mean: 30.00150015666294 usec\nrounds: 9565"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 25781.640370561552,
            "unit": "iter/sec",
            "range": "stddev: 0.000004441676666914806",
            "extra": "mean: 38.78729148444091 usec\nrounds: 11215"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 33427.8134484352,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017154540023154346",
            "extra": "mean: 29.915208230492603 usec\nrounds: 8529"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 23994.544185174684,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022950034776263566",
            "extra": "mean: 41.67614072110034 usec\nrounds: 10567"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 27934.980647945704,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023128972852162468",
            "extra": "mean: 35.79741158952757 usec\nrounds: 8732"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 17770.435108929913,
            "unit": "iter/sec",
            "range": "stddev: 0.000004135376056821911",
            "extra": "mean: 56.27324226278988 usec\nrounds: 7690"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 9900.085608568279,
            "unit": "iter/sec",
            "range": "stddev: 0.0000071890618295425736",
            "extra": "mean: 101.00922754996428 usec\nrounds: 3775"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 60303.52695203143,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018283018794154096",
            "extra": "mean: 16.582777998962683 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 29742.27574489424,
            "unit": "iter/sec",
            "range": "stddev: 0.000001968677742466664",
            "extra": "mean: 33.622175000232346 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 50707.96174063888,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011500379670482137",
            "extra": "mean: 19.72076900102593 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 25252.049544602258,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021396051965699236",
            "extra": "mean: 39.6007460001897 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 755.1263657253317,
            "unit": "iter/sec",
            "range": "stddev: 0.00002179964590441978",
            "extra": "mean: 1.324281663823851 msec\nrounds: 586"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 978.9206006762814,
            "unit": "iter/sec",
            "range": "stddev: 0.00009357622767623823",
            "extra": "mean: 1.0215333085330474 msec\nrounds: 914"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6052.320123345539,
            "unit": "iter/sec",
            "range": "stddev: 0.000007417715683528935",
            "extra": "mean: 165.22589347888464 usec\nrounds: 2469"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 6663.764876639864,
            "unit": "iter/sec",
            "range": "stddev: 0.000006665306160632681",
            "extra": "mean: 150.06531870677884 usec\nrounds: 3897"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 6841.715527113327,
            "unit": "iter/sec",
            "range": "stddev: 0.000009911573705402392",
            "extra": "mean: 146.1621717589773 usec\nrounds: 2160"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4517.6000400109215,
            "unit": "iter/sec",
            "range": "stddev: 0.000016574619498190774",
            "extra": "mean: 221.3564705027722 usec\nrounds: 2763"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 2059.0285529316848,
            "unit": "iter/sec",
            "range": "stddev: 0.000023278407165957396",
            "extra": "mean: 485.6659217164234 usec\nrounds: 1865"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 2064.4327759770244,
            "unit": "iter/sec",
            "range": "stddev: 0.00002387806103517635",
            "extra": "mean: 484.3945570117848 usec\nrounds: 1833"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 2537.07848266652,
            "unit": "iter/sec",
            "range": "stddev: 0.000014843380229893383",
            "extra": "mean: 394.1541449474516 usec\nrounds: 2187"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 2528.146405477973,
            "unit": "iter/sec",
            "range": "stddev: 0.000019600371734153325",
            "extra": "mean: 395.5467127351509 usec\nrounds: 2395"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1797.7655597252208,
            "unit": "iter/sec",
            "range": "stddev: 0.000022613381594984244",
            "extra": "mean: 556.246054770815 usec\nrounds: 1698"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1801.777371527533,
            "unit": "iter/sec",
            "range": "stddev: 0.00001727814366194996",
            "extra": "mean: 555.0075252372649 usec\nrounds: 1367"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 2182.5692480725043,
            "unit": "iter/sec",
            "range": "stddev: 0.000021670449213696505",
            "extra": "mean: 458.1756115564863 usec\nrounds: 1869"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 2185.6863164049655,
            "unit": "iter/sec",
            "range": "stddev: 0.000023907958431929612",
            "extra": "mean: 457.52219451362447 usec\nrounds: 2005"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 595.1726737081209,
            "unit": "iter/sec",
            "range": "stddev: 0.00002590614784777114",
            "extra": "mean: 1.6801846660225042 msec\nrounds: 518"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 606.4479668707511,
            "unit": "iter/sec",
            "range": "stddev: 0.000016147966812539057",
            "extra": "mean: 1.6489460838000045 msec\nrounds: 537"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1f78271024cac9398133f42d017c931d36a3ee18",
          "message": "[Feature] Expose call_on_nested to apply and named_apply (#768)",
          "timestamp": "2024-04-30T18:04:16+01:00",
          "tree_id": "5798046ad043a7fc0f83618efa3d17a42fe4afda",
          "url": "https://github.com/pytorch/tensordict/commit/1f78271024cac9398133f42d017c931d36a3ee18"
        },
        "date": 1714497204710,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 60741.47212046508,
            "unit": "iter/sec",
            "range": "stddev: 7.736911789376499e-7",
            "extra": "mean: 16.46321640043161 usec\nrounds: 8036"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 60478.01102130116,
            "unit": "iter/sec",
            "range": "stddev: 8.028026537212143e-7",
            "extra": "mean: 16.534935311410734 usec\nrounds: 17623"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 53200.56020495283,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013727269833190266",
            "extra": "mean: 18.796794547793176 usec\nrounds: 31657"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 53414.69574922469,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010475792758668629",
            "extra": "mean: 18.72143959585345 usec\nrounds: 34948"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 386818.2994614605,
            "unit": "iter/sec",
            "range": "stddev: 1.9147156430083451e-7",
            "extra": "mean: 2.585193103305166 usec\nrounds: 103328"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 3863.692986439948,
            "unit": "iter/sec",
            "range": "stddev: 0.000005626531293639109",
            "extra": "mean: 258.8197363272934 usec\nrounds: 2651"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 3810.951079906126,
            "unit": "iter/sec",
            "range": "stddev: 0.000014712793254879803",
            "extra": "mean: 262.40168898327414 usec\nrounds: 3704"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 13111.87908317543,
            "unit": "iter/sec",
            "range": "stddev: 0.000002413619377425246",
            "extra": "mean: 76.26671918315314 usec\nrounds: 8867"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 3772.1689059480773,
            "unit": "iter/sec",
            "range": "stddev: 0.000025494603387221105",
            "extra": "mean: 265.09947590712807 usec\nrounds: 3528"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 12585.601629925199,
            "unit": "iter/sec",
            "range": "stddev: 0.000005006128193690702",
            "extra": "mean: 79.4558758019376 usec\nrounds: 10910"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 3799.272121744749,
            "unit": "iter/sec",
            "range": "stddev: 0.000005206193418453266",
            "extra": "mean: 263.20831147540116 usec\nrounds: 3721"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 263125.17445238796,
            "unit": "iter/sec",
            "range": "stddev: 2.653971800891155e-7",
            "extra": "mean: 3.8004725396617203 usec\nrounds: 120265"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 7211.349821764804,
            "unit": "iter/sec",
            "range": "stddev: 0.000005502967880488989",
            "extra": "mean: 138.67029401095868 usec\nrounds: 6095"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 6928.916245005286,
            "unit": "iter/sec",
            "range": "stddev: 0.000021836633780256285",
            "extra": "mean: 144.32271435245744 usec\nrounds: 6466"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 8463.454963631655,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028911053019902694",
            "extra": "mean: 118.15505656934478 usec\nrounds: 7566"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 7130.797290086828,
            "unit": "iter/sec",
            "range": "stddev: 0.000006755444117946239",
            "extra": "mean: 140.23677287674286 usec\nrounds: 6437"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 8463.02877462202,
            "unit": "iter/sec",
            "range": "stddev: 0.000002976977852832339",
            "extra": "mean: 118.16100673067399 usec\nrounds: 7428"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 6974.114081710391,
            "unit": "iter/sec",
            "range": "stddev: 0.000003921629087734678",
            "extra": "mean: 143.38738774326896 usec\nrounds: 6494"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 866275.584083361,
            "unit": "iter/sec",
            "range": "stddev: 5.830429198128086e-8",
            "extra": "mean: 1.1543670609834205 usec\nrounds: 162549"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 19686.265147122096,
            "unit": "iter/sec",
            "range": "stddev: 0.000002912879701720392",
            "extra": "mean: 50.796836907694924 usec\nrounds: 15200"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 19686.036763039825,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016545846948905003",
            "extra": "mean: 50.797426218236154 usec\nrounds: 17667"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 21767.992761139933,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015381151798243497",
            "extra": "mean: 45.93900829410386 usec\nrounds: 17965"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 19175.922090731783,
            "unit": "iter/sec",
            "range": "stddev: 0.000003255255754334409",
            "extra": "mean: 52.14873085468603 usec\nrounds: 15761"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 22026.295551695814,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015644961703436185",
            "extra": "mean: 45.4002806624017 usec\nrounds: 17815"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 19362.514777976096,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016581116095851625",
            "extra": "mean: 51.64618395217189 usec\nrounds: 17423"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 754905.5855495328,
            "unit": "iter/sec",
            "range": "stddev: 1.5169755753531603e-7",
            "extra": "mean: 1.3246689640957563 usec\nrounds: 163372"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 292539.27770793956,
            "unit": "iter/sec",
            "range": "stddev: 2.887122834849441e-7",
            "extra": "mean: 3.4183443940760774 usec\nrounds: 86491"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 291754.1192640205,
            "unit": "iter/sec",
            "range": "stddev: 3.496342495841601e-7",
            "extra": "mean: 3.4275437225105914 usec\nrounds: 115261"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 295217.80580211454,
            "unit": "iter/sec",
            "range": "stddev: 3.042561996301864e-7",
            "extra": "mean: 3.3873295592146744 usec\nrounds: 81348"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 294261.46942090435,
            "unit": "iter/sec",
            "range": "stddev: 2.756310274747958e-7",
            "extra": "mean: 3.39833822609519 usec\nrounds: 56616"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 220031.04162806307,
            "unit": "iter/sec",
            "range": "stddev: 0.000001454855200948547",
            "extra": "mean: 4.544813279984303 usec\nrounds: 95603"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 241834.91608835652,
            "unit": "iter/sec",
            "range": "stddev: 3.78605809998715e-7",
            "extra": "mean: 4.135052192523933 usec\nrounds: 109326"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 243607.43934787062,
            "unit": "iter/sec",
            "range": "stddev: 3.8452449910648704e-7",
            "extra": "mean: 4.104964949662326 usec\nrounds: 74322"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 241695.30446969086,
            "unit": "iter/sec",
            "range": "stddev: 5.919153962798713e-7",
            "extra": "mean: 4.137440742566856 usec\nrounds: 77197"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 95638.88044842749,
            "unit": "iter/sec",
            "range": "stddev: 5.721684774813518e-7",
            "extra": "mean: 10.455998599222855 usec\nrounds: 71393"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 101447.0218211512,
            "unit": "iter/sec",
            "range": "stddev: 5.35896649758266e-7",
            "extra": "mean: 9.857361823425208 usec\nrounds: 77676"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 97222.45059794807,
            "unit": "iter/sec",
            "range": "stddev: 4.916959314488902e-7",
            "extra": "mean: 10.285690124551392 usec\nrounds: 59695"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 102191.0293725781,
            "unit": "iter/sec",
            "range": "stddev: 5.244464692716017e-7",
            "extra": "mean: 9.785594744858688 usec\nrounds: 61539"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 92822.71553470062,
            "unit": "iter/sec",
            "range": "stddev: 5.813450970299262e-7",
            "extra": "mean: 10.773225004672078 usec\nrounds: 53150"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 99194.04440248685,
            "unit": "iter/sec",
            "range": "stddev: 9.606205607609356e-7",
            "extra": "mean: 10.08125040191354 usec\nrounds: 68414"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 92274.47624110073,
            "unit": "iter/sec",
            "range": "stddev: 5.441914280692662e-7",
            "extra": "mean: 10.837233011078116 usec\nrounds: 56139"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 98788.2837436388,
            "unit": "iter/sec",
            "range": "stddev: 5.695770800422587e-7",
            "extra": "mean: 10.122657891244033 usec\nrounds: 61346"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2843.36131233799,
            "unit": "iter/sec",
            "range": "stddev: 0.000160777191297778",
            "extra": "mean: 351.6964219991223 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3232.962593445264,
            "unit": "iter/sec",
            "range": "stddev: 0.000011399784492807577",
            "extra": "mean: 309.3138170009979 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2510.896184652078,
            "unit": "iter/sec",
            "range": "stddev: 0.0015592824106082188",
            "extra": "mean: 398.26417599920205 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3140.6128107011814,
            "unit": "iter/sec",
            "range": "stddev: 0.000011306737839986708",
            "extra": "mean: 318.40919599915196 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 10450.484602875598,
            "unit": "iter/sec",
            "range": "stddev: 0.000006307444554693087",
            "extra": "mean: 95.6893424564097 usec\nrounds: 7563"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 2465.2442010552945,
            "unit": "iter/sec",
            "range": "stddev: 0.000013805554606651132",
            "extra": "mean: 405.63932756516823 usec\nrounds: 2311"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1409.3409042302135,
            "unit": "iter/sec",
            "range": "stddev: 0.0001194660461124992",
            "extra": "mean: 709.5515336271342 usec\nrounds: 907"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 531139.7074281892,
            "unit": "iter/sec",
            "range": "stddev: 2.587747865095702e-7",
            "extra": "mean: 1.8827438167672699 usec\nrounds: 124286"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 99978.90343910687,
            "unit": "iter/sec",
            "range": "stddev: 6.434707531179836e-7",
            "extra": "mean: 10.002110101248107 usec\nrounds: 20808"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 78869.58559811422,
            "unit": "iter/sec",
            "range": "stddev: 9.139318792002177e-7",
            "extra": "mean: 12.67915879633974 usec\nrounds: 23760"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 62473.961096882405,
            "unit": "iter/sec",
            "range": "stddev: 9.383457992306893e-7",
            "extra": "mean: 16.00666873754388 usec\nrounds: 23338"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 73364.03957382205,
            "unit": "iter/sec",
            "range": "stddev: 0.0000038433273342735175",
            "extra": "mean: 13.630656188087313 usec\nrounds: 14348"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 84677.25585051208,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032459965191472847",
            "extra": "mean: 11.809546612674653 usec\nrounds: 11778"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 43731.53785178211,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022340692976203778",
            "extra": "mean: 22.86679245969506 usec\nrounds: 11405"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 15464.842229757725,
            "unit": "iter/sec",
            "range": "stddev: 0.00001600367477828595",
            "extra": "mean: 64.66279999131075 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 53416.379246825134,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014924655823095899",
            "extra": "mean: 18.72084956150292 usec\nrounds: 15940"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 23822.175938450284,
            "unit": "iter/sec",
            "range": "stddev: 0.0000040617996756538175",
            "extra": "mean: 41.97769349801274 usec\nrounds: 7305"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 27794.203381709292,
            "unit": "iter/sec",
            "range": "stddev: 0.0000050941490289899685",
            "extra": "mean: 35.97872499767618 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 15900.933368885451,
            "unit": "iter/sec",
            "range": "stddev: 0.00000501977524045032",
            "extra": "mean: 62.889390000009364 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 11632.749582100549,
            "unit": "iter/sec",
            "range": "stddev: 0.0000059170946106019035",
            "extra": "mean: 85.96419900061392 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 19684.220227960694,
            "unit": "iter/sec",
            "range": "stddev: 0.000004755315273936135",
            "extra": "mean: 50.802113998884124 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 49205.355852969114,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017804192256246393",
            "extra": "mean: 20.32299091562527 usec\nrounds: 15631"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 51093.19860020101,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020524311101201843",
            "extra": "mean: 19.572076663762953 usec\nrounds: 19240"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 7024.034129950897,
            "unit": "iter/sec",
            "range": "stddev: 0.00006567238870326167",
            "extra": "mean: 142.3683287266417 usec\nrounds: 3763"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 46762.00825720611,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021368629139667016",
            "extra": "mean: 21.38488138703706 usec\nrounds: 25149"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 33743.87172693383,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023536746201560284",
            "extra": "mean: 29.635010709272454 usec\nrounds: 17369"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 39491.040611189885,
            "unit": "iter/sec",
            "range": "stddev: 0.000002298408597468293",
            "extra": "mean: 25.32219927667967 usec\nrounds: 10503"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 46450.25739866058,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023245951705436327",
            "extra": "mean: 21.528405998215966 usec\nrounds: 16872"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 39248.68060173598,
            "unit": "iter/sec",
            "range": "stddev: 0.000002416744224147201",
            "extra": "mean: 25.478563474456507 usec\nrounds: 15195"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 25039.96543420534,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028589519995713285",
            "extra": "mean: 39.93615736521625 usec\nrounds: 10733"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 16742.67431965117,
            "unit": "iter/sec",
            "range": "stddev: 0.000004299137091936999",
            "extra": "mean: 59.727614651518515 usec\nrounds: 11452"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 8277.682154542577,
            "unit": "iter/sec",
            "range": "stddev: 0.000004203207251989812",
            "extra": "mean: 120.80676466312808 usec\nrounds: 5541"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2530.9186278331194,
            "unit": "iter/sec",
            "range": "stddev: 0.000009850503745046021",
            "extra": "mean: 395.11345366965185 usec\nrounds: 2180"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 902315.4092435115,
            "unit": "iter/sec",
            "range": "stddev: 7.893426944432398e-8",
            "extra": "mean: 1.108259916383772 usec\nrounds: 172981"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3777.9804630703115,
            "unit": "iter/sec",
            "range": "stddev: 0.000026525973071334725",
            "extra": "mean: 264.6916811177245 usec\nrounds: 2537"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 3909.9386907567464,
            "unit": "iter/sec",
            "range": "stddev: 0.000008674217360746905",
            "extra": "mean: 255.75848602537954 usec\nrounds: 3399"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1299.618651854479,
            "unit": "iter/sec",
            "range": "stddev: 0.0028108924688621455",
            "extra": "mean: 769.4564852336177 usec\nrounds: 1422"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 618.4157423502634,
            "unit": "iter/sec",
            "range": "stddev: 0.0024277989927227518",
            "extra": "mean: 1.6170351618145125 msec\nrounds: 618"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 612.4499960463316,
            "unit": "iter/sec",
            "range": "stddev: 0.0027037387584674033",
            "extra": "mean: 1.63278636044656 msec\nrounds: 627"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 9853.496168097161,
            "unit": "iter/sec",
            "range": "stddev: 0.000007340997348267987",
            "extra": "mean: 101.48682081368415 usec\nrounds: 2556"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 12250.41367626198,
            "unit": "iter/sec",
            "range": "stddev: 0.00002945713816485729",
            "extra": "mean: 81.62989646118906 usec\nrounds: 9127"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 179938.4278311449,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014384666140469452",
            "extra": "mean: 5.557456581416867 usec\nrounds: 23481"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1586557.1836137865,
            "unit": "iter/sec",
            "range": "stddev: 1.1370366422650197e-7",
            "extra": "mean: 630.2955924489569 nsec\nrounds: 73121"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 254679.7473280204,
            "unit": "iter/sec",
            "range": "stddev: 5.1104680636408e-7",
            "extra": "mean: 3.92649989051555 usec\nrounds: 22857"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 4195.5000223177085,
            "unit": "iter/sec",
            "range": "stddev: 0.00005086043215111981",
            "extra": "mean: 238.35061248493875 usec\nrounds: 1778"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 3212.424439822189,
            "unit": "iter/sec",
            "range": "stddev: 0.00005283437631803202",
            "extra": "mean: 311.2913684766236 usec\nrounds: 2798"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1657.05809220517,
            "unit": "iter/sec",
            "range": "stddev: 0.000056198281128985616",
            "extra": "mean: 603.4791445779827 usec\nrounds: 1411"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 8.9824291918021,
            "unit": "iter/sec",
            "range": "stddev: 0.024807271395686198",
            "extra": "mean: 111.32845900000632 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 2.6309121677771254,
            "unit": "iter/sec",
            "range": "stddev: 0.08509192544127842",
            "extra": "mean: 380.0963073749841 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 8.978915788510484,
            "unit": "iter/sec",
            "range": "stddev: 0.022711897017914956",
            "extra": "mean: 111.37202125000556 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 8.108056551513895,
            "unit": "iter/sec",
            "range": "stddev: 0.0030849900202725876",
            "extra": "mean: 123.33411757140311 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 2.3504109141870475,
            "unit": "iter/sec",
            "range": "stddev: 0.07355716899049373",
            "extra": "mean: 425.4575206250166 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_filesystem",
            "value": 9.673826798263338,
            "unit": "iter/sec",
            "range": "stddev: 0.019919444082351065",
            "extra": "mean: 103.37170809999634 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_filesystem",
            "value": 10.633689638496394,
            "unit": "iter/sec",
            "range": "stddev: 0.004524784783991039",
            "extra": "mean: 94.04073600002118 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 39306.253309608524,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013786323797693855",
            "extra": "mean: 25.441244478917234 usec\nrounds: 9600"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 29972.446826308995,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019306398420215287",
            "extra": "mean: 33.363976114296655 usec\nrounds: 8080"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 39310.36241438366,
            "unit": "iter/sec",
            "range": "stddev: 0.000001340156312189559",
            "extra": "mean: 25.438585110426256 usec\nrounds: 14346"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 26072.968145888168,
            "unit": "iter/sec",
            "range": "stddev: 0.000001981666240869075",
            "extra": "mean: 38.35389950252767 usec\nrounds: 9463"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 33806.70394987026,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015606142898253686",
            "extra": "mean: 29.579931882233602 usec\nrounds: 11392"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 25647.478597439847,
            "unit": "iter/sec",
            "range": "stddev: 0.000005226183824676285",
            "extra": "mean: 38.99018752275402 usec\nrounds: 10868"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 33703.0347654424,
            "unit": "iter/sec",
            "range": "stddev: 0.00000224822666734032",
            "extra": "mean: 29.67091856740912 usec\nrounds: 7847"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 24343.673022742896,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021146326151337654",
            "extra": "mean: 41.07843541382426 usec\nrounds: 10397"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 27739.99643497199,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023681929719733044",
            "extra": "mean: 36.04903130914947 usec\nrounds: 8496"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 17286.08963479677,
            "unit": "iter/sec",
            "range": "stddev: 0.000004945446155152321",
            "extra": "mean: 57.84998349117706 usec\nrounds: 6966"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 9846.36939887558,
            "unit": "iter/sec",
            "range": "stddev: 0.000008110700038389681",
            "extra": "mean: 101.56027663497942 usec\nrounds: 3745"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 58578.50585569042,
            "unit": "iter/sec",
            "range": "stddev: 8.568346469502025e-7",
            "extra": "mean: 17.071108001005086 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 29134.55455880883,
            "unit": "iter/sec",
            "range": "stddev: 0.000003069817414016088",
            "extra": "mean: 34.323504002145455 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 50326.38169079499,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015034598708840352",
            "extra": "mean: 19.870293996973487 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 25794.17558184636,
            "unit": "iter/sec",
            "range": "stddev: 0.000002306685915531363",
            "extra": "mean: 38.76844200067353 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 756.1609632154809,
            "unit": "iter/sec",
            "range": "stddev: 0.00003448287795484536",
            "extra": "mean: 1.3224697500220373 msec\nrounds: 16"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 985.0405292274751,
            "unit": "iter/sec",
            "range": "stddev: 0.00009866010839729786",
            "extra": "mean: 1.0151866550955593 msec\nrounds: 922"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6212.189307352166,
            "unit": "iter/sec",
            "range": "stddev: 0.000006914027861305916",
            "extra": "mean: 160.97384521371453 usec\nrounds: 2371"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 6657.699918948286,
            "unit": "iter/sec",
            "range": "stddev: 0.000007951420344325305",
            "extra": "mean: 150.2020235477915 usec\nrounds: 3992"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 6660.421979501804,
            "unit": "iter/sec",
            "range": "stddev: 0.000009907468135242693",
            "extra": "mean: 150.14063719650377 usec\nrounds: 1742"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4499.097978512739,
            "unit": "iter/sec",
            "range": "stddev: 0.00001854412544383208",
            "extra": "mean: 222.26677542385255 usec\nrounds: 2596"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 2059.023351925298,
            "unit": "iter/sec",
            "range": "stddev: 0.000019088379227411678",
            "extra": "mean: 485.66714848811506 usec\nrounds: 1852"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 2066.430045772515,
            "unit": "iter/sec",
            "range": "stddev: 0.000021565803182914804",
            "extra": "mean: 483.9263743990712 usec\nrounds: 1867"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 2523.4189032138283,
            "unit": "iter/sec",
            "range": "stddev: 0.00001609383318050931",
            "extra": "mean: 396.2877502131728 usec\nrounds: 2350"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 2519.7273879734744,
            "unit": "iter/sec",
            "range": "stddev: 0.000017720161002772095",
            "extra": "mean: 396.8683297935115 usec\nrounds: 2259"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1810.594623618061,
            "unit": "iter/sec",
            "range": "stddev: 0.000028259916118419937",
            "extra": "mean: 552.3047439529715 usec\nrounds: 1695"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1812.0454603403566,
            "unit": "iter/sec",
            "range": "stddev: 0.00002472976480233433",
            "extra": "mean: 551.8625342943493 usec\nrounds: 1662"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 2193.7691183915827,
            "unit": "iter/sec",
            "range": "stddev: 0.00002113957171472245",
            "extra": "mean: 455.8364832545256 usec\nrounds: 2090"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 2184.7478726752397,
            "unit": "iter/sec",
            "range": "stddev: 0.000020270059527063113",
            "extra": "mean: 457.71872008988055 usec\nrounds: 1822"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 591.6192377281837,
            "unit": "iter/sec",
            "range": "stddev: 0.00007952005764916654",
            "extra": "mean: 1.690276340302924 msec\nrounds: 526"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 606.1662680642816,
            "unit": "iter/sec",
            "range": "stddev: 0.00003149164055373112",
            "extra": "mean: 1.6497123853383968 msec\nrounds: 532"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f691a35702c7b6adbc8b2de281d3934a7fed6e7b",
          "message": "[Feature] Load tensordicts on device, incl. meta (#769)",
          "timestamp": "2024-05-01T09:15:06+01:00",
          "tree_id": "e7d87de85d99dc55d50dc8604b80248ef007a739",
          "url": "https://github.com/pytorch/tensordict/commit/f691a35702c7b6adbc8b2de281d3934a7fed6e7b"
        },
        "date": 1714551569176,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 55692.69303913596,
            "unit": "iter/sec",
            "range": "stddev: 6.552237730167515e-7",
            "extra": "mean: 17.95567686585541 usec\nrounds: 8105"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 54082.1133681239,
            "unit": "iter/sec",
            "range": "stddev: 0.000002074192333316219",
            "extra": "mean: 18.490401682220536 usec\nrounds: 17118"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 49346.44923141183,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010577595445946993",
            "extra": "mean: 20.264882591865252 usec\nrounds: 26540"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 49329.98150052941,
            "unit": "iter/sec",
            "range": "stddev: 9.815333627999379e-7",
            "extra": "mean: 20.271647577837996 usec\nrounds: 31607"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 367833.9343073567,
            "unit": "iter/sec",
            "range": "stddev: 3.3803699340079955e-7",
            "extra": "mean: 2.718618122830436 usec\nrounds: 101441"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 3747.221546099115,
            "unit": "iter/sec",
            "range": "stddev: 0.000006321932274535111",
            "extra": "mean: 266.86439210967046 usec\nrounds: 3244"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 3759.730590287973,
            "unit": "iter/sec",
            "range": "stddev: 0.000008818365420609425",
            "extra": "mean: 265.9765044291128 usec\nrounds: 3499"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 12949.630304540013,
            "unit": "iter/sec",
            "range": "stddev: 0.000006492339262095812",
            "extra": "mean: 77.22228175497874 usec\nrounds: 10282"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 3750.278623417441,
            "unit": "iter/sec",
            "range": "stddev: 0.0000048102903836443726",
            "extra": "mean: 266.64685491787543 usec\nrounds: 3467"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 13173.104218612018,
            "unit": "iter/sec",
            "range": "stddev: 0.000001998234792744258",
            "extra": "mean: 75.91225146363905 usec\nrounds: 11274"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 3706.3914177776173,
            "unit": "iter/sec",
            "range": "stddev: 0.000005891613327034635",
            "extra": "mean: 269.80420772709647 usec\nrounds: 3649"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 245123.43947903093,
            "unit": "iter/sec",
            "range": "stddev: 4.2201776883364355e-7",
            "extra": "mean: 4.079577220870161 usec\nrounds: 111285"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 7191.030152260837,
            "unit": "iter/sec",
            "range": "stddev: 0.000012845180213693267",
            "extra": "mean: 139.0621341902736 usec\nrounds: 6066"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 6940.642809537328,
            "unit": "iter/sec",
            "range": "stddev: 0.000024418488177357495",
            "extra": "mean: 144.07887387979002 usec\nrounds: 6470"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 8494.220238746002,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026566839186417797",
            "extra": "mean: 117.72710995160512 usec\nrounds: 7567"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 7253.741459154615,
            "unit": "iter/sec",
            "range": "stddev: 0.000006003433621759348",
            "extra": "mean: 137.85989004859636 usec\nrounds: 6612"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 8615.617542644419,
            "unit": "iter/sec",
            "range": "stddev: 0.000006009277770852859",
            "extra": "mean: 116.06829052593564 usec\nrounds: 7452"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 7098.906846578556,
            "unit": "iter/sec",
            "range": "stddev: 0.000005086576651392098",
            "extra": "mean: 140.86675901120856 usec\nrounds: 6631"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 820071.2944955016,
            "unit": "iter/sec",
            "range": "stddev: 2.1928675643179925e-7",
            "extra": "mean: 1.219406174453635 usec\nrounds: 196503"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 19354.42323769829,
            "unit": "iter/sec",
            "range": "stddev: 0.000006423886407224627",
            "extra": "mean: 51.66777576984125 usec\nrounds: 14610"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 19301.812927899373,
            "unit": "iter/sec",
            "range": "stddev: 0.000006633958042093411",
            "extra": "mean: 51.80860490853543 usec\nrounds: 17601"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 21630.70800384579,
            "unit": "iter/sec",
            "range": "stddev: 0.000004299104133943023",
            "extra": "mean: 46.23057182512042 usec\nrounds: 17654"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 19317.551989444357,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016994397452063671",
            "extra": "mean: 51.76639361687379 usec\nrounds: 15416"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 22073.004175391976,
            "unit": "iter/sec",
            "range": "stddev: 0.000002574349313425516",
            "extra": "mean: 45.30420925280515 usec\nrounds: 17682"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 19309.78701617215,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018247593617981684",
            "extra": "mean: 51.787210245379164 usec\nrounds: 17570"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 718746.230568691,
            "unit": "iter/sec",
            "range": "stddev: 1.9905583941565067e-7",
            "extra": "mean: 1.3913116444572846 usec\nrounds: 147864"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 282650.56969210994,
            "unit": "iter/sec",
            "range": "stddev: 3.6281248922450315e-7",
            "extra": "mean: 3.5379373234212674 usec\nrounds: 108015"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 278674.4058727246,
            "unit": "iter/sec",
            "range": "stddev: 2.918475412734114e-7",
            "extra": "mean: 3.588417087921297 usec\nrounds: 108850"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 277691.4088361363,
            "unit": "iter/sec",
            "range": "stddev: 2.8610403789889654e-7",
            "extra": "mean: 3.6011196896267426 usec\nrounds: 76782"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 283661.59930590144,
            "unit": "iter/sec",
            "range": "stddev: 2.991107970408767e-7",
            "extra": "mean: 3.525327370525036 usec\nrounds: 47237"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 228481.8043496019,
            "unit": "iter/sec",
            "range": "stddev: 3.594672565671629e-7",
            "extra": "mean: 4.37671613652828 usec\nrounds: 100513"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 226963.86858496477,
            "unit": "iter/sec",
            "range": "stddev: 4.1723972830611523e-7",
            "extra": "mean: 4.405987641269193 usec\nrounds: 99523"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 227963.96911318507,
            "unit": "iter/sec",
            "range": "stddev: 3.757379311992331e-7",
            "extra": "mean: 4.386658136766762 usec\nrounds: 66278"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 227589.81788008087,
            "unit": "iter/sec",
            "range": "stddev: 4.411921266618405e-7",
            "extra": "mean: 4.393869678857553 usec\nrounds: 73503"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 92254.92644642752,
            "unit": "iter/sec",
            "range": "stddev: 4.805926855384408e-7",
            "extra": "mean: 10.83952953537609 usec\nrounds: 69848"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 97329.62513527935,
            "unit": "iter/sec",
            "range": "stddev: 6.052699136382584e-7",
            "extra": "mean: 10.274364034692322 usec\nrounds: 74655"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 93261.7142705311,
            "unit": "iter/sec",
            "range": "stddev: 4.837958612959182e-7",
            "extra": "mean: 10.722513604020046 usec\nrounds: 56456"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 98113.96976086532,
            "unit": "iter/sec",
            "range": "stddev: 6.677257178660782e-7",
            "extra": "mean: 10.19222851177376 usec\nrounds: 60754"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 87700.02633062568,
            "unit": "iter/sec",
            "range": "stddev: 5.774567431658379e-7",
            "extra": "mean: 11.402505128448183 usec\nrounds: 51185"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 95602.20169367493,
            "unit": "iter/sec",
            "range": "stddev: 4.931768620265478e-7",
            "extra": "mean: 10.460010149182164 usec\nrounds: 68181"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 88623.50972291293,
            "unit": "iter/sec",
            "range": "stddev: 7.831527087798389e-7",
            "extra": "mean: 11.283687625626246 usec\nrounds: 53692"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 95647.08283426375,
            "unit": "iter/sec",
            "range": "stddev: 4.993684543152912e-7",
            "extra": "mean: 10.455101926451738 usec\nrounds: 57561"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2819.552510571759,
            "unit": "iter/sec",
            "range": "stddev: 0.0001421193190280767",
            "extra": "mean: 354.6662089996744 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3265.4764816990723,
            "unit": "iter/sec",
            "range": "stddev: 0.000012806566178183734",
            "extra": "mean: 306.234022999206 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2477.9487096501134,
            "unit": "iter/sec",
            "range": "stddev: 0.0016782008377656088",
            "extra": "mean: 403.55960400052027 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3178.814387171483,
            "unit": "iter/sec",
            "range": "stddev: 0.000013832234220497415",
            "extra": "mean: 314.5826960000022 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 10376.901407565765,
            "unit": "iter/sec",
            "range": "stddev: 0.000007031191951148414",
            "extra": "mean: 96.36788100067166 usec\nrounds: 7353"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 2376.969130982332,
            "unit": "iter/sec",
            "range": "stddev: 0.00001872155623309317",
            "extra": "mean: 420.7038227655608 usec\nrounds: 2082"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1347.5629299793973,
            "unit": "iter/sec",
            "range": "stddev: 0.00015818542565639864",
            "extra": "mean: 742.0803717235593 usec\nrounds: 877"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 531891.2624794933,
            "unit": "iter/sec",
            "range": "stddev: 1.9159081272283393e-7",
            "extra": "mean: 1.8800835256032324 usec\nrounds: 97192"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 85825.86178053866,
            "unit": "iter/sec",
            "range": "stddev: 5.295537061672678e-7",
            "extra": "mean: 11.651499667513432 usec\nrounds: 15080"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 69735.4767421939,
            "unit": "iter/sec",
            "range": "stddev: 6.566353123117024e-7",
            "extra": "mean: 14.339903399483658 usec\nrounds: 16180"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 56592.17722496353,
            "unit": "iter/sec",
            "range": "stddev: 9.913402616361639e-7",
            "extra": "mean: 17.670286761098268 usec\nrounds: 20198"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 74620.40316316717,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012989767138463967",
            "extra": "mean: 13.401160508518972 usec\nrounds: 13937"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 86715.37380172359,
            "unit": "iter/sec",
            "range": "stddev: 8.635954628415165e-7",
            "extra": "mean: 11.531980503094177 usec\nrounds: 9745"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 43006.782875243276,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019286766995962004",
            "extra": "mean: 23.252146130085144 usec\nrounds: 10518"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 16934.686308345717,
            "unit": "iter/sec",
            "range": "stddev: 0.000013573621599695609",
            "extra": "mean: 59.050399977422785 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 53028.369185183044,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015751682738284306",
            "extra": "mean: 18.857830541758688 usec\nrounds: 15048"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 23635.359707983822,
            "unit": "iter/sec",
            "range": "stddev: 0.000003912536342130286",
            "extra": "mean: 42.30948935641579 usec\nrounds: 7375"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 27934.10790795129,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024047214436754004",
            "extra": "mean: 35.79853000121602 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 15951.666450425424,
            "unit": "iter/sec",
            "range": "stddev: 0.000004385558782597785",
            "extra": "mean: 62.68937500090032 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 11521.42409989664,
            "unit": "iter/sec",
            "range": "stddev: 0.000006307158889371857",
            "extra": "mean: 86.79482599802668 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 19483.349879302274,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034308663500563914",
            "extra": "mean: 51.325876001556026 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 47716.80420239561,
            "unit": "iter/sec",
            "range": "stddev: 0.000002539770142698221",
            "extra": "mean: 20.956977666785896 usec\nrounds: 15806"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 49056.89760598298,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020223448090480796",
            "extra": "mean: 20.384493288422707 usec\nrounds: 18030"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 6862.528496032502,
            "unit": "iter/sec",
            "range": "stddev: 0.0000735466646224821",
            "extra": "mean: 145.71888489470598 usec\nrounds: 3588"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 42956.68546250691,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028722971582223043",
            "extra": "mean: 23.27926350073754 usec\nrounds: 24110"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 31734.449382874904,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034134012065243686",
            "extra": "mean: 31.51149679438388 usec\nrounds: 16379"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 39348.74672849325,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036648266368069434",
            "extra": "mean: 25.413770021698788 usec\nrounds: 12049"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 43436.99864371196,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030638724452275926",
            "extra": "mean: 23.021848452339196 usec\nrounds: 15058"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 37018.366909198114,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033024154296388127",
            "extra": "mean: 27.01361738763051 usec\nrounds: 13941"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 23908.264371793695,
            "unit": "iter/sec",
            "range": "stddev: 0.000004550999174568232",
            "extra": "mean: 41.82654100059944 usec\nrounds: 8573"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 16276.026500934184,
            "unit": "iter/sec",
            "range": "stddev: 0.00000408856460601274",
            "extra": "mean: 61.44005724877651 usec\nrounds: 10795"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 8044.313730080595,
            "unit": "iter/sec",
            "range": "stddev: 0.000010706402706294175",
            "extra": "mean: 124.31141220420567 usec\nrounds: 5342"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2501.964050660164,
            "unit": "iter/sec",
            "range": "stddev: 0.000010170733338497508",
            "extra": "mean: 399.68599858025203 usec\nrounds: 2113"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 923023.933706328,
            "unit": "iter/sec",
            "range": "stddev: 9.467671545279761e-8",
            "extra": "mean: 1.0833955258176104 usec\nrounds: 174826"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3778.7443078611605,
            "unit": "iter/sec",
            "range": "stddev: 0.00010315196900783551",
            "extra": "mean: 264.6381756817038 usec\nrounds: 2385"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 3935.6341527596383,
            "unit": "iter/sec",
            "range": "stddev: 0.00001534749079293045",
            "extra": "mean: 254.088657935547 usec\nrounds: 3245"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1294.829186440562,
            "unit": "iter/sec",
            "range": "stddev: 0.003038524107368418",
            "extra": "mean: 772.3026407436516 usec\nrounds: 1453"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 618.0559889637755,
            "unit": "iter/sec",
            "range": "stddev: 0.0025918262103748163",
            "extra": "mean: 1.6179763934924194 msec\nrounds: 615"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 618.1339004772337,
            "unit": "iter/sec",
            "range": "stddev: 0.0026173314156372985",
            "extra": "mean: 1.617772458730939 msec\nrounds: 630"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 9541.389234837236,
            "unit": "iter/sec",
            "range": "stddev: 0.000008470225672430672",
            "extra": "mean: 104.80654078641189 usec\nrounds: 2513"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 11862.081837929201,
            "unit": "iter/sec",
            "range": "stddev: 0.00004557615637893567",
            "extra": "mean: 84.3022340987805 usec\nrounds: 8411"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 174279.94855384715,
            "unit": "iter/sec",
            "range": "stddev: 0.000001789694837246227",
            "extra": "mean: 5.737894739457252 usec\nrounds: 21233"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1559061.9618030312,
            "unit": "iter/sec",
            "range": "stddev: 1.1082808377426537e-7",
            "extra": "mean: 641.4113258484707 nsec\nrounds: 73020"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 264039.2518140425,
            "unit": "iter/sec",
            "range": "stddev: 4.2646491682586894e-7",
            "extra": "mean: 3.787315685564356 usec\nrounds: 24233"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 3994.377871374892,
            "unit": "iter/sec",
            "range": "stddev: 0.000054765769362603006",
            "extra": "mean: 250.35187761437132 usec\nrounds: 1626"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 3071.540765055195,
            "unit": "iter/sec",
            "range": "stddev: 0.0000537633002925892",
            "extra": "mean: 325.56950289475657 usec\nrounds: 2764"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1534.3989730509884,
            "unit": "iter/sec",
            "range": "stddev: 0.00005526855436002068",
            "extra": "mean: 651.7209784177625 usec\nrounds: 1390"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 8.67695087373604,
            "unit": "iter/sec",
            "range": "stddev: 0.027442123559239886",
            "extra": "mean: 115.24785774999202 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 2.609064647933115,
            "unit": "iter/sec",
            "range": "stddev: 0.08218303550666345",
            "extra": "mean: 383.2791191250067 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.281071285431693,
            "unit": "iter/sec",
            "range": "stddev: 0.024343885334773947",
            "extra": "mean: 107.74618244444254 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 7.9405795902305965,
            "unit": "iter/sec",
            "range": "stddev: 0.00471725398840818",
            "extra": "mean: 125.93539157145578 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 2.491498106072706,
            "unit": "iter/sec",
            "range": "stddev: 0.09115345433374897",
            "extra": "mean: 401.36494487498453 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_filesystem",
            "value": 10.484376374714811,
            "unit": "iter/sec",
            "range": "stddev: 0.006774602181024125",
            "extra": "mean: 95.38001729999905 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_filesystem",
            "value": 9.968375080848084,
            "unit": "iter/sec",
            "range": "stddev: 0.02255567283619774",
            "extra": "mean: 100.31725249998544 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 39121.91533442881,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013921926796841866",
            "extra": "mean: 25.561120703105274 usec\nrounds: 9163"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 28974.74913960069,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024750955467570763",
            "extra": "mean: 34.512809590929955 usec\nrounds: 7715"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 39458.08128747848,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016630246551737751",
            "extra": "mean: 25.34335090229887 usec\nrounds: 13964"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 26073.32674747035,
            "unit": "iter/sec",
            "range": "stddev: 0.000002329120536553519",
            "extra": "mean: 38.35337199910712 usec\nrounds: 9285"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 34021.06128918311,
            "unit": "iter/sec",
            "range": "stddev: 0.000002067736240055934",
            "extra": "mean: 29.393556876426626 usec\nrounds: 10347"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 25713.916067038987,
            "unit": "iter/sec",
            "range": "stddev: 0.000005365846511794205",
            "extra": "mean: 38.889447931341564 usec\nrounds: 11187"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 34329.783148643866,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025852216491360707",
            "extra": "mean: 29.12922565429905 usec\nrounds: 7835"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 24590.100076201892,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029201844616019087",
            "extra": "mean: 40.666772274253255 usec\nrounds: 9753"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 28024.009265568468,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026776939762095813",
            "extra": "mean: 35.68368788789419 usec\nrounds: 8058"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 16750.04048154381,
            "unit": "iter/sec",
            "range": "stddev: 0.000004813462514231429",
            "extra": "mean: 59.70134825058241 usec\nrounds: 7374"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 9554.857834323933,
            "unit": "iter/sec",
            "range": "stddev: 0.00000963737964427543",
            "extra": "mean: 104.65880469803518 usec\nrounds: 3661"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 55777.77110074104,
            "unit": "iter/sec",
            "range": "stddev: 0.00000151778940465163",
            "extra": "mean: 17.92828899874621 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 27542.54531304189,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023199944240408045",
            "extra": "mean: 36.30746500130044 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 47505.44400687691,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019934728410129813",
            "extra": "mean: 21.050218999221215 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 23928.739637954066,
            "unit": "iter/sec",
            "range": "stddev: 0.000002172963938519995",
            "extra": "mean: 41.79075100194041 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 751.5363644085209,
            "unit": "iter/sec",
            "range": "stddev: 0.000026238084904681937",
            "extra": "mean: 1.330607602450517 msec\nrounds: 571"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 963.9204903926013,
            "unit": "iter/sec",
            "range": "stddev: 0.0001072983819575638",
            "extra": "mean: 1.0374299643663594 msec\nrounds: 898"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6041.999362180093,
            "unit": "iter/sec",
            "range": "stddev: 0.000007537614360208708",
            "extra": "mean: 165.50812736914568 usec\nrounds: 2481"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 6501.80264683705,
            "unit": "iter/sec",
            "range": "stddev: 0.000007745005086951354",
            "extra": "mean: 153.80349947817515 usec\nrounds: 3832"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 6573.092469872264,
            "unit": "iter/sec",
            "range": "stddev: 0.000008506237392709034",
            "extra": "mean: 152.13539206750778 usec\nrounds: 1992"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4349.688257826853,
            "unit": "iter/sec",
            "range": "stddev: 0.00002042827346152763",
            "extra": "mean: 229.9015333341635 usec\nrounds: 2385"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 2035.067956749179,
            "unit": "iter/sec",
            "range": "stddev: 0.00002716605579597203",
            "extra": "mean: 491.3840821303097 usec\nrounds: 1802"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 2045.7005762170406,
            "unit": "iter/sec",
            "range": "stddev: 0.000021963403332600926",
            "extra": "mean: 488.8300915714774 usec\nrounds: 1922"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 2518.68497768006,
            "unit": "iter/sec",
            "range": "stddev: 0.000018273320266075015",
            "extra": "mean: 397.03258202663034 usec\nrounds: 2426"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 2512.1919377727318,
            "unit": "iter/sec",
            "range": "stddev: 0.000018825048706283676",
            "extra": "mean: 398.0587569620909 usec\nrounds: 2370"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1770.949642595361,
            "unit": "iter/sec",
            "range": "stddev: 0.000029240707461798387",
            "extra": "mean: 564.6687946103767 usec\nrounds: 1558"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1778.8982098987328,
            "unit": "iter/sec",
            "range": "stddev: 0.00001925377044311512",
            "extra": "mean: 562.1457115620611 usec\nrounds: 1574"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 2184.1635531346337,
            "unit": "iter/sec",
            "range": "stddev: 0.00002191543166708502",
            "extra": "mean: 457.8411715390249 usec\nrounds: 2087"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 2179.072449131395,
            "unit": "iter/sec",
            "range": "stddev: 0.00002661830790828147",
            "extra": "mean: 458.9108546614006 usec\nrounds: 2071"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 592.9509316932858,
            "unit": "iter/sec",
            "range": "stddev: 0.00004055844817367313",
            "extra": "mean: 1.6864801900965176 msec\nrounds: 505"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 600.7922442355282,
            "unit": "iter/sec",
            "range": "stddev: 0.00008899901215591196",
            "extra": "mean: 1.6644688901942126 msec\nrounds: 510"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "59cd9a289dedc20bc95182304ed54ebfcf577cfe",
          "message": "[BugFix] Fix device parsing in augmented funcs (#770)",
          "timestamp": "2024-05-02T15:46:23+01:00",
          "tree_id": "c9e92ce28da05c4a543e283b4370decf070f9e0c",
          "url": "https://github.com/pytorch/tensordict/commit/59cd9a289dedc20bc95182304ed54ebfcf577cfe"
        },
        "date": 1714661448233,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 51295.56768380327,
            "unit": "iter/sec",
            "range": "stddev: 0.000005319937017882057",
            "extra": "mean: 19.49486174252348 usec\nrounds: 7826"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 55788.18967729691,
            "unit": "iter/sec",
            "range": "stddev: 9.558262107630782e-7",
            "extra": "mean: 17.92494084831277 usec\nrounds: 16483"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 49551.06379908304,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013884702866685916",
            "extra": "mean: 20.18120143807095 usec\nrounds: 30183"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 49891.73438079024,
            "unit": "iter/sec",
            "range": "stddev: 9.128024059398846e-7",
            "extra": "mean: 20.043400222723644 usec\nrounds: 33294"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 392561.0416535396,
            "unit": "iter/sec",
            "range": "stddev: 2.5509803942487933e-7",
            "extra": "mean: 2.547374532602154 usec\nrounds: 108602"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 3757.1957176253886,
            "unit": "iter/sec",
            "range": "stddev: 0.000009253750218510539",
            "extra": "mean: 266.1559511815948 usec\nrounds: 3175"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 3784.583426595016,
            "unit": "iter/sec",
            "range": "stddev: 0.000021616040561564508",
            "extra": "mean: 264.2298734843054 usec\nrounds: 3549"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 12987.763042692803,
            "unit": "iter/sec",
            "range": "stddev: 0.000004937860688646939",
            "extra": "mean: 76.99555317669748 usec\nrounds: 10183"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 3720.047334325563,
            "unit": "iter/sec",
            "range": "stddev: 0.00002164839983170978",
            "extra": "mean: 268.8137838389355 usec\nrounds: 3465"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 12966.762058601571,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020624872960467205",
            "extra": "mean: 77.12025527117964 usec\nrounds: 11004"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 3711.435827845835,
            "unit": "iter/sec",
            "range": "stddev: 0.000019829741137707767",
            "extra": "mean: 269.43750246125444 usec\nrounds: 3453"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 254822.95269091014,
            "unit": "iter/sec",
            "range": "stddev: 2.389244750466412e-7",
            "extra": "mean: 3.924293276724406 usec\nrounds: 110169"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 7221.103440089446,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032351004843875394",
            "extra": "mean: 138.48299062554534 usec\nrounds: 6080"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 6901.376368428357,
            "unit": "iter/sec",
            "range": "stddev: 0.000022881502458428376",
            "extra": "mean: 144.89863276761545 usec\nrounds: 6421"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 8419.53732358976,
            "unit": "iter/sec",
            "range": "stddev: 0.000006170341563495307",
            "extra": "mean: 118.77137205606438 usec\nrounds: 7558"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 7314.21072902781,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032810087139681707",
            "extra": "mean: 136.7201516400551 usec\nrounds: 6555"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 8419.417505805623,
            "unit": "iter/sec",
            "range": "stddev: 0.000003927661687889033",
            "extra": "mean: 118.77306230632327 usec\nrounds: 7431"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 6989.921019276621,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034523656231804414",
            "extra": "mean: 143.06313293701405 usec\nrounds: 6552"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 852833.7389818354,
            "unit": "iter/sec",
            "range": "stddev: 6.455768247942937e-8",
            "extra": "mean: 1.1725614903484711 usec\nrounds: 193799"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 19570.155512723133,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018896958101038932",
            "extra": "mean: 51.09821428602704 usec\nrounds: 14952"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 19300.576577654992,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023184655554084613",
            "extra": "mean: 51.811923647801166 usec\nrounds: 15900"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 21786.423350216523,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022775082263842232",
            "extra": "mean: 45.90014542199106 usec\nrounds: 10858"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 19142.58315481172,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017999868468488867",
            "extra": "mean: 52.23955366486878 usec\nrounds: 15345"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 21653.219273548075,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017759201282633578",
            "extra": "mean: 46.18250927803683 usec\nrounds: 17892"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 19257.040671595012,
            "unit": "iter/sec",
            "range": "stddev: 0.000002306296528202264",
            "extra": "mean: 51.929058937651014 usec\nrounds: 17561"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 741855.2895940999,
            "unit": "iter/sec",
            "range": "stddev: 1.7904607351735642e-7",
            "extra": "mean: 1.3479717864479228 usec\nrounds: 145922"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 290409.63496243674,
            "unit": "iter/sec",
            "range": "stddev: 2.894710413810368e-7",
            "extra": "mean: 3.443411924433381 usec\nrounds: 111285"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 289804.441120805,
            "unit": "iter/sec",
            "range": "stddev: 2.5981843780531827e-7",
            "extra": "mean: 3.4506027448459635 usec\nrounds: 118400"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 290719.9107863689,
            "unit": "iter/sec",
            "range": "stddev: 2.910068392552567e-7",
            "extra": "mean: 3.4397368838449967 usec\nrounds: 68274"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 289879.0397597477,
            "unit": "iter/sec",
            "range": "stddev: 3.6735863270404335e-7",
            "extra": "mean: 3.4497147528458836 usec\nrounds: 51664"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 235647.96540173638,
            "unit": "iter/sec",
            "range": "stddev: 3.630620808977029e-7",
            "extra": "mean: 4.243618222186575 usec\nrounds: 103445"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 233992.42860922677,
            "unit": "iter/sec",
            "range": "stddev: 5.208690000148783e-7",
            "extra": "mean: 4.273642553067497 usec\nrounds: 102902"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 145550.35315086285,
            "unit": "iter/sec",
            "range": "stddev: 7.422717308237039e-7",
            "extra": "mean: 6.870474570154431 usec\nrounds: 52340"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 145541.80700314516,
            "unit": "iter/sec",
            "range": "stddev: 5.951417792084316e-7",
            "extra": "mean: 6.870878001249428 usec\nrounds: 59099"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 88752.40261223857,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024443888923656364",
            "extra": "mean: 11.267300608965197 usec\nrounds: 69459"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 96278.23050565433,
            "unit": "iter/sec",
            "range": "stddev: 9.45734244637463e-7",
            "extra": "mean: 10.386563969320884 usec\nrounds: 72855"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 90859.8328198415,
            "unit": "iter/sec",
            "range": "stddev: 6.827280099728041e-7",
            "extra": "mean: 11.005963460034291 usec\nrounds: 56076"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 96583.48866414292,
            "unit": "iter/sec",
            "range": "stddev: 5.021747644172574e-7",
            "extra": "mean: 10.35373658407987 usec\nrounds: 61196"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 86157.33538629749,
            "unit": "iter/sec",
            "range": "stddev: 5.860067550636164e-7",
            "extra": "mean: 11.606672786668382 usec\nrounds: 51798"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 93696.64622124411,
            "unit": "iter/sec",
            "range": "stddev: 5.953962092765972e-7",
            "extra": "mean: 10.672740597766104 usec\nrounds: 64479"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 86454.695705961,
            "unit": "iter/sec",
            "range": "stddev: 8.048421087546409e-7",
            "extra": "mean: 11.5667517170042 usec\nrounds: 52122"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 95575.82114405898,
            "unit": "iter/sec",
            "range": "stddev: 5.184414643268859e-7",
            "extra": "mean: 10.462897289605555 usec\nrounds: 57297"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2827.9915405879246,
            "unit": "iter/sec",
            "range": "stddev: 0.00013967152085114632",
            "extra": "mean: 353.60784699946635 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3270.995229625101,
            "unit": "iter/sec",
            "range": "stddev: 0.000012255275514715279",
            "extra": "mean: 305.71735199828254 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2494.116727809588,
            "unit": "iter/sec",
            "range": "stddev: 0.0015645928380502293",
            "extra": "mean: 400.94354400093835 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3165.7535667869392,
            "unit": "iter/sec",
            "range": "stddev: 0.00001944033373938778",
            "extra": "mean: 315.8805569995593 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 10328.053722400773,
            "unit": "iter/sec",
            "range": "stddev: 0.000005093210687908292",
            "extra": "mean: 96.82366367160495 usec\nrounds: 7790"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 2392.2056536896844,
            "unit": "iter/sec",
            "range": "stddev: 0.00002167090182596903",
            "extra": "mean: 418.02426077274015 usec\nrounds: 2251"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1330.1683239290667,
            "unit": "iter/sec",
            "range": "stddev: 0.00013556244838729782",
            "extra": "mean: 751.7845538872767 usec\nrounds: 798"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 532690.0953090976,
            "unit": "iter/sec",
            "range": "stddev: 1.6777292541867133e-7",
            "extra": "mean: 1.877264114362296 usec\nrounds: 125866"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 81755.21736551299,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013420427012631482",
            "extra": "mean: 12.231635267131374 usec\nrounds: 19310"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 67216.92207086385,
            "unit": "iter/sec",
            "range": "stddev: 8.248675978440028e-7",
            "extra": "mean: 14.877206054537039 usec\nrounds: 24081"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 54369.719570468675,
            "unit": "iter/sec",
            "range": "stddev: 9.262365246517903e-7",
            "extra": "mean: 18.392590726974387 usec\nrounds: 21438"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 73919.16354117694,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020183047327367425",
            "extra": "mean: 13.528291610645004 usec\nrounds: 727"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 81778.26816328986,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015232899703307025",
            "extra": "mean: 12.228187542480857 usec\nrounds: 11880"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 42507.06898892371,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017513158665862375",
            "extra": "mean: 23.52549878846211 usec\nrounds: 11969"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 16849.710687259252,
            "unit": "iter/sec",
            "range": "stddev: 0.000011285454041381354",
            "extra": "mean: 59.34820001129992 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 50409.73435674423,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018926227192799137",
            "extra": "mean: 19.837438398764576 usec\nrounds: 15714"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 24006.703740051933,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033912358901032573",
            "extra": "mean: 41.65503147904622 usec\nrounds: 7497"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 25888.22902033973,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033041234648244856",
            "extra": "mean: 38.62759400089999 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 15324.23154644404,
            "unit": "iter/sec",
            "range": "stddev: 0.0000039362129159591454",
            "extra": "mean: 65.25612700181682 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 11353.363247163363,
            "unit": "iter/sec",
            "range": "stddev: 0.0000038069047144605706",
            "extra": "mean: 88.07962699950167 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 18730.83355797281,
            "unit": "iter/sec",
            "range": "stddev: 0.000003925647600744512",
            "extra": "mean: 53.387906998636936 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 47746.99337410628,
            "unit": "iter/sec",
            "range": "stddev: 0.000002295220652725279",
            "extra": "mean: 20.943727119419233 usec\nrounds: 16597"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 49072.10550613791,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016106015248880383",
            "extra": "mean: 20.378175945088 usec\nrounds: 17892"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 7019.972697998601,
            "unit": "iter/sec",
            "range": "stddev: 0.00009155820946948031",
            "extra": "mean: 142.45069646568578 usec\nrounds: 3848"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 42790.09879581887,
            "unit": "iter/sec",
            "range": "stddev: 0.000002097033630980729",
            "extra": "mean: 23.369892291478248 usec\nrounds: 22598"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 30923.370126900547,
            "unit": "iter/sec",
            "range": "stddev: 0.000002139475772153994",
            "extra": "mean: 32.33800183797205 usec\nrounds: 15777"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 39729.28067057934,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019165784860563875",
            "extra": "mean: 25.17035252391389 usec\nrounds: 12107"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 43745.71418173891,
            "unit": "iter/sec",
            "range": "stddev: 0.000001629785962450289",
            "extra": "mean: 22.859382197889392 usec\nrounds: 16447"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 37365.23315407177,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032494174766357876",
            "extra": "mean: 26.76284651768666 usec\nrounds: 15207"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 23494.90589628787,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027074301613454456",
            "extra": "mean: 42.562417760438755 usec\nrounds: 10968"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 16251.352700846963,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022231601300195162",
            "extra": "mean: 61.53333931075679 usec\nrounds: 10819"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 8229.416290758998,
            "unit": "iter/sec",
            "range": "stddev: 0.000003935552488116704",
            "extra": "mean: 121.51530128849153 usec\nrounds: 5045"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2375.050606133401,
            "unit": "iter/sec",
            "range": "stddev: 0.00008610850390129962",
            "extra": "mean: 421.04366004563036 usec\nrounds: 2165"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 885897.673875381,
            "unit": "iter/sec",
            "range": "stddev: 2.625714381135485e-7",
            "extra": "mean: 1.128798539029316 usec\nrounds: 184163"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3718.568470615789,
            "unit": "iter/sec",
            "range": "stddev: 0.000027965134257044044",
            "extra": "mean: 268.92069028767986 usec\nrounds: 2399"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 3936.1520921994606,
            "unit": "iter/sec",
            "range": "stddev: 0.000027134920834140413",
            "extra": "mean: 254.05522362354031 usec\nrounds: 3412"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1326.6185792918461,
            "unit": "iter/sec",
            "range": "stddev: 0.00295073362311229",
            "extra": "mean: 753.7961668935798 usec\nrounds: 1462"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 613.5403535776819,
            "unit": "iter/sec",
            "range": "stddev: 0.0028876705725546384",
            "extra": "mean: 1.6298846427440203 msec\nrounds: 627"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 618.6968270636731,
            "unit": "iter/sec",
            "range": "stddev: 0.002636490030021322",
            "extra": "mean: 1.6163005146575369 msec\nrounds: 614"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 9787.657653489287,
            "unit": "iter/sec",
            "range": "stddev: 0.000006465095450867382",
            "extra": "mean: 102.16949094490461 usec\nrounds: 2650"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 12048.830683626158,
            "unit": "iter/sec",
            "range": "stddev: 0.00004371845849739145",
            "extra": "mean: 82.99560565316574 usec\nrounds: 8703"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 171055.4830474157,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021747904488342385",
            "extra": "mean: 5.846056391672667 usec\nrounds: 23514"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1582073.6045402992,
            "unit": "iter/sec",
            "range": "stddev: 1.5410509452071495e-7",
            "extra": "mean: 632.0818431773082 nsec\nrounds: 72749"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 273663.02583931905,
            "unit": "iter/sec",
            "range": "stddev: 3.6037303914680955e-7",
            "extra": "mean: 3.654129003847048 usec\nrounds: 27999"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 3937.5151819690427,
            "unit": "iter/sec",
            "range": "stddev: 0.00005794623741855914",
            "extra": "mean: 253.96727473693895 usec\nrounds: 1718"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 3051.802934569137,
            "unit": "iter/sec",
            "range": "stddev: 0.00005510411090747471",
            "extra": "mean: 327.67515512635254 usec\nrounds: 2585"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1585.5256572353253,
            "unit": "iter/sec",
            "range": "stddev: 0.00006196999234733575",
            "extra": "mean: 630.70565615677 usec\nrounds: 1364"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 8.810960864390037,
            "unit": "iter/sec",
            "range": "stddev: 0.02268746674407983",
            "extra": "mean: 113.49499962501852 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 2.489745697579759,
            "unit": "iter/sec",
            "range": "stddev: 0.1071042213836284",
            "extra": "mean: 401.6474457500152 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.289230574657635,
            "unit": "iter/sec",
            "range": "stddev: 0.022648908400866453",
            "extra": "mean: 107.65154249999398 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 7.452441708259141,
            "unit": "iter/sec",
            "range": "stddev: 0.03259512559691175",
            "extra": "mean: 134.18420957144204 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 2.4832837911240064,
            "unit": "iter/sec",
            "range": "stddev: 0.09607880958948596",
            "extra": "mean: 402.6925974285729 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_filesystem",
            "value": 10.8334546752477,
            "unit": "iter/sec",
            "range": "stddev: 0.003279728394647762",
            "extra": "mean: 92.30665840000256 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_filesystem",
            "value": 9.836189503893873,
            "unit": "iter/sec",
            "range": "stddev: 0.01901632269155221",
            "extra": "mean: 101.66538572728066 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 38286.90765351611,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019292210510123166",
            "extra": "mean: 26.118588867235516 usec\nrounds: 9126"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 28606.050583253225,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015482375402790966",
            "extra": "mean: 34.95763936687673 usec\nrounds: 7900"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 38436.078975657125,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013244790649590779",
            "extra": "mean: 26.017222012508974 usec\nrounds: 14319"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 25744.36550708308,
            "unit": "iter/sec",
            "range": "stddev: 0.000001897328256158495",
            "extra": "mean: 38.843450996097324 usec\nrounds: 9836"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 32855.28439603528,
            "unit": "iter/sec",
            "range": "stddev: 0.000039331777613198496",
            "extra": "mean: 30.436504153976284 usec\nrounds: 10592"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 25658.481713585985,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036106157716829084",
            "extra": "mean: 38.97346737669623 usec\nrounds: 11265"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 33579.691695833055,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016770420482203413",
            "extra": "mean: 29.77990414736569 usec\nrounds: 8367"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 23438.513578198814,
            "unit": "iter/sec",
            "range": "stddev: 0.000003831305608786742",
            "extra": "mean: 42.66482158365809 usec\nrounds: 9618"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 28056.748281335633,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020577931536763398",
            "extra": "mean: 35.642049106069656 usec\nrounds: 8390"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 17373.87224532079,
            "unit": "iter/sec",
            "range": "stddev: 0.000003775979871473367",
            "extra": "mean: 57.557692716966116 usec\nrounds: 7703"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 9721.75751510347,
            "unit": "iter/sec",
            "range": "stddev: 0.000008182457262569349",
            "extra": "mean: 102.86205950379095 usec\nrounds: 3025"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 52473.08261903306,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019542889285654412",
            "extra": "mean: 19.057390000511987 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 26726.958407684808,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016694337880882494",
            "extra": "mean: 37.415406001173324 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 45397.93423115678,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012991162355616403",
            "extra": "mean: 22.02743399971041 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 23262.68347917211,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020685008216496223",
            "extra": "mean: 42.9873019978686 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 765.2712656795286,
            "unit": "iter/sec",
            "range": "stddev: 0.00002436422337938122",
            "extra": "mean: 1.3067261830509762 msec\nrounds: 590"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 977.9979761417862,
            "unit": "iter/sec",
            "range": "stddev: 0.00011207679470717641",
            "extra": "mean: 1.0224970034651932 msec\nrounds: 866"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6348.274851678085,
            "unit": "iter/sec",
            "range": "stddev: 0.000006527524642301149",
            "extra": "mean: 157.5231103511 usec\nrounds: 2483"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 6767.312788818911,
            "unit": "iter/sec",
            "range": "stddev: 0.00000653749582199886",
            "extra": "mean: 147.76914134251632 usec\nrounds: 3962"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 6961.4081538676755,
            "unit": "iter/sec",
            "range": "stddev: 0.000007531924726667215",
            "extra": "mean: 143.64909769647852 usec\nrounds: 2170"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4584.387852692368,
            "unit": "iter/sec",
            "range": "stddev: 0.00001841139671417172",
            "extra": "mean: 218.13163112119958 usec\nrounds: 2654"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 2062.869680111141,
            "unit": "iter/sec",
            "range": "stddev: 0.000020119688940104623",
            "extra": "mean: 484.7615967413526 usec\nrounds: 1902"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 2068.280782774165,
            "unit": "iter/sec",
            "range": "stddev: 0.00001918651700698975",
            "extra": "mean: 483.4933478706454 usec\nrounds: 1949"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 2570.6085046947474,
            "unit": "iter/sec",
            "range": "stddev: 0.00001656644455048413",
            "extra": "mean: 389.012950892243 usec\nrounds: 2403"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 2559.1010165631646,
            "unit": "iter/sec",
            "range": "stddev: 0.000022674107345642832",
            "extra": "mean: 390.76222217401386 usec\nrounds: 2372"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1781.93566491013,
            "unit": "iter/sec",
            "range": "stddev: 0.00002347799402313138",
            "extra": "mean: 561.187488242138 usec\nrounds: 1616"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1801.4283908261546,
            "unit": "iter/sec",
            "range": "stddev: 0.00002483180008997243",
            "extra": "mean: 555.1150437577977 usec\nrounds: 1714"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 2209.831357178869,
            "unit": "iter/sec",
            "range": "stddev: 0.000021884822533840985",
            "extra": "mean: 452.52321936305003 usec\nrounds: 1983"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 2209.9046303077644,
            "unit": "iter/sec",
            "range": "stddev: 0.000020742907652208122",
            "extra": "mean: 452.50821518969076 usec\nrounds: 1896"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 593.3290156084777,
            "unit": "iter/sec",
            "range": "stddev: 0.00008128835244782253",
            "extra": "mean: 1.685405523231437 msec\nrounds: 495"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 603.4340529331716,
            "unit": "iter/sec",
            "range": "stddev: 0.00002091943771966171",
            "extra": "mean: 1.6571819159677863 msec\nrounds: 476"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1b84e0c68d9f892e4407f788a0aa8250af8ce20d",
          "message": "[Feature] Support for non tensor data in h5 (#772)",
          "timestamp": "2024-05-08T19:40:13+01:00",
          "tree_id": "579fd14eab72bd343a6d4de8f4507eb9549c040a",
          "url": "https://github.com/pytorch/tensordict/commit/1b84e0c68d9f892e4407f788a0aa8250af8ce20d"
        },
        "date": 1715194114920,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 60054.11856550987,
            "unit": "iter/sec",
            "range": "stddev: 6.57797794030104e-7",
            "extra": "mean: 16.651647278931467 usec\nrounds: 11760"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 59141.34393761724,
            "unit": "iter/sec",
            "range": "stddev: 8.613100099495804e-7",
            "extra": "mean: 16.908645178148266 usec\nrounds: 17296"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 52925.040417171906,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010402399754991255",
            "extra": "mean: 18.89464782865887 usec\nrounds: 31868"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 53371.02536373977,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011391798179441302",
            "extra": "mean: 18.73675825384084 usec\nrounds: 33651"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 346763.7912260904,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030527569796045563",
            "extra": "mean: 2.883807436941993 usec\nrounds: 132206"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 3756.914459040451,
            "unit": "iter/sec",
            "range": "stddev: 0.000012837314125403331",
            "extra": "mean: 266.17587674737973 usec\nrounds: 3075"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 3754.497852398089,
            "unit": "iter/sec",
            "range": "stddev: 0.000021276967479763167",
            "extra": "mean: 266.3472025589989 usec\nrounds: 3673"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 12654.359642695023,
            "unit": "iter/sec",
            "range": "stddev: 0.000016151311049034123",
            "extra": "mean: 79.02414884954449 usec\nrounds: 9298"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 3729.808368124528,
            "unit": "iter/sec",
            "range": "stddev: 0.000012117756981931085",
            "extra": "mean: 268.11028913606987 usec\nrounds: 3507"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 13118.080651885075,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022860487035676196",
            "extra": "mean: 76.23066411444111 usec\nrounds: 11361"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 3729.008484343624,
            "unit": "iter/sec",
            "range": "stddev: 0.000011914289752064803",
            "extra": "mean: 268.16779961711967 usec\nrounds: 3653"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 259823.74287700743,
            "unit": "iter/sec",
            "range": "stddev: 2.8805155623607784e-7",
            "extra": "mean: 3.8487629687998495 usec\nrounds: 108838"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 7252.721998085496,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033491978588813155",
            "extra": "mean: 137.87926798572596 usec\nrounds: 5963"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 7031.401063989282,
            "unit": "iter/sec",
            "range": "stddev: 0.000028136279318799162",
            "extra": "mean: 142.21916669231317 usec\nrounds: 6503"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 8512.885505114324,
            "unit": "iter/sec",
            "range": "stddev: 0.00000994915780618248",
            "extra": "mean: 117.46898268503972 usec\nrounds: 7508"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 7279.149848930706,
            "unit": "iter/sec",
            "range": "stddev: 0.00001088939530005104",
            "extra": "mean: 137.37868030658805 usec\nrounds: 6647"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 8593.337762881316,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032625590559637466",
            "extra": "mean: 116.3692185264115 usec\nrounds: 7697"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 7172.588069221895,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033229181659124757",
            "extra": "mean: 139.41968928775847 usec\nrounds: 6749"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 867739.5276468707,
            "unit": "iter/sec",
            "range": "stddev: 2.1384929309503297e-7",
            "extra": "mean: 1.1524195546465614 usec\nrounds: 165536"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 19736.385363179244,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015025491969565198",
            "extra": "mean: 50.667839201479524 usec\nrounds: 14969"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 19835.80401218502,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016995595797876653",
            "extra": "mean: 50.413887906217745 usec\nrounds: 18092"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 21825.77879951775,
            "unit": "iter/sec",
            "range": "stddev: 0.00000287589765890709",
            "extra": "mean: 45.81737995173375 usec\nrounds: 16081"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 19341.62992792594,
            "unit": "iter/sec",
            "range": "stddev: 0.000002170869769905569",
            "extra": "mean: 51.701950855557136 usec\nrounds: 15831"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 22206.71427038699,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016311769805052658",
            "extra": "mean: 45.03142553302071 usec\nrounds: 17162"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 19323.476830810785,
            "unit": "iter/sec",
            "range": "stddev: 0.000004757487283941012",
            "extra": "mean: 51.75052133503873 usec\nrounds: 17741"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 724717.9499612729,
            "unit": "iter/sec",
            "range": "stddev: 2.656932574979889e-7",
            "extra": "mean: 1.379847153024756 usec\nrounds: 138812"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 286333.3671079633,
            "unit": "iter/sec",
            "range": "stddev: 4.529747006348435e-7",
            "extra": "mean: 3.492432649747542 usec\nrounds: 103756"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 288968.8418194295,
            "unit": "iter/sec",
            "range": "stddev: 2.980202073516303e-7",
            "extra": "mean: 3.4605807107220192 usec\nrounds: 116199"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 291068.1395039677,
            "unit": "iter/sec",
            "range": "stddev: 2.7625854043947856e-7",
            "extra": "mean: 3.435621644142087 usec\nrounds: 82285"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 289636.90332275286,
            "unit": "iter/sec",
            "range": "stddev: 4.879360502601149e-7",
            "extra": "mean: 3.4525987142103363 usec\nrounds: 86791"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 236704.04427078442,
            "unit": "iter/sec",
            "range": "stddev: 3.082736644267154e-7",
            "extra": "mean: 4.224684893241711 usec\nrounds: 104298"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 236394.33058371834,
            "unit": "iter/sec",
            "range": "stddev: 4.000116465011587e-7",
            "extra": "mean: 4.2302198937290205 usec\nrounds: 106747"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 235662.77716237243,
            "unit": "iter/sec",
            "range": "stddev: 3.434120544279959e-7",
            "extra": "mean: 4.243351504387121 usec\nrounds: 75211"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 236941.37877226507,
            "unit": "iter/sec",
            "range": "stddev: 4.384016951010203e-7",
            "extra": "mean: 4.2204531989372125 usec\nrounds: 76366"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 94356.4423859133,
            "unit": "iter/sec",
            "range": "stddev: 4.798428801926228e-7",
            "extra": "mean: 10.598110470401675 usec\nrounds: 69358"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 99847.64357653617,
            "unit": "iter/sec",
            "range": "stddev: 7.451398672925134e-7",
            "extra": "mean: 10.015258890245821 usec\nrounds: 76600"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 93192.13593450977,
            "unit": "iter/sec",
            "range": "stddev: 8.118213407458456e-7",
            "extra": "mean: 10.730519157783274 usec\nrounds: 58096"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 99565.24299000438,
            "unit": "iter/sec",
            "range": "stddev: 7.38567433441876e-7",
            "extra": "mean: 10.043665539995645 usec\nrounds: 61535"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 90236.01004984585,
            "unit": "iter/sec",
            "range": "stddev: 5.922496998362613e-7",
            "extra": "mean: 11.082050275135236 usec\nrounds: 53605"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 98456.95263439472,
            "unit": "iter/sec",
            "range": "stddev: 8.789997266832797e-7",
            "extra": "mean: 10.156723047414962 usec\nrounds: 65105"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 90390.60862268548,
            "unit": "iter/sec",
            "range": "stddev: 8.558553554557112e-7",
            "extra": "mean: 11.063096213615143 usec\nrounds: 45638"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 98627.02748452105,
            "unit": "iter/sec",
            "range": "stddev: 4.925662337254148e-7",
            "extra": "mean: 10.139208546632355 usec\nrounds: 57661"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2898.4314688889463,
            "unit": "iter/sec",
            "range": "stddev: 0.00017326674259430302",
            "extra": "mean: 345.01419499952135 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3341.8090857297457,
            "unit": "iter/sec",
            "range": "stddev: 0.00001111011800647816",
            "extra": "mean: 299.2391170010933 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2490.765846783559,
            "unit": "iter/sec",
            "range": "stddev: 0.001837375865810116",
            "extra": "mean: 401.4829420000865 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3252.24677729549,
            "unit": "iter/sec",
            "range": "stddev: 0.000013712738896639187",
            "extra": "mean: 307.47974199903183 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 10469.630330631722,
            "unit": "iter/sec",
            "range": "stddev: 0.000006537660111259384",
            "extra": "mean: 95.51435613483227 usec\nrounds: 7792"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 2464.1815330246554,
            "unit": "iter/sec",
            "range": "stddev: 0.00001055570770844217",
            "extra": "mean: 405.8142578369832 usec\nrounds: 2265"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1338.0466814349145,
            "unit": "iter/sec",
            "range": "stddev: 0.00018651818744046234",
            "extra": "mean: 747.3580809061199 usec\nrounds: 927"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 532370.0669551027,
            "unit": "iter/sec",
            "range": "stddev: 1.7636781050115412e-7",
            "extra": "mean: 1.8783926108383826 usec\nrounds: 112651"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 97412.10771738144,
            "unit": "iter/sec",
            "range": "stddev: 5.786470528169332e-7",
            "extra": "mean: 10.26566433508725 usec\nrounds: 20160"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 76896.23063641344,
            "unit": "iter/sec",
            "range": "stddev: 8.611987435536895e-7",
            "extra": "mean: 13.004538606427607 usec\nrounds: 22820"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 61595.577730062374,
            "unit": "iter/sec",
            "range": "stddev: 8.629857498717028e-7",
            "extra": "mean: 16.23493174108081 usec\nrounds: 22649"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 75422.76026928777,
            "unit": "iter/sec",
            "range": "stddev: 0.000001417668552029218",
            "extra": "mean: 13.258597224891027 usec\nrounds: 13551"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 88839.25046167138,
            "unit": "iter/sec",
            "range": "stddev: 6.43580125009425e-7",
            "extra": "mean: 11.25628587367965 usec\nrounds: 11680"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 44132.25996597916,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019165799157131074",
            "extra": "mean: 22.659161365651425 usec\nrounds: 8930"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 16115.671850915744,
            "unit": "iter/sec",
            "range": "stddev: 0.000016387575530860095",
            "extra": "mean: 62.05139998201048 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 52984.7911315402,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013225281742481082",
            "extra": "mean: 18.873340417958747 usec\nrounds: 15763"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 23964.967170761378,
            "unit": "iter/sec",
            "range": "stddev: 0.00000351989860777567",
            "extra": "mean: 41.72757646086229 usec\nrounds: 7239"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 29135.356720456733,
            "unit": "iter/sec",
            "range": "stddev: 0.000002986575795607619",
            "extra": "mean: 34.32255899917891 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 16654.379620219635,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037249175514026248",
            "extra": "mean: 60.04426600111401 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 12131.160555211341,
            "unit": "iter/sec",
            "range": "stddev: 0.000003918833561536724",
            "extra": "mean: 82.43234399947141 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 20130.18836879683,
            "unit": "iter/sec",
            "range": "stddev: 0.000004036003775391527",
            "extra": "mean: 49.676634002594255 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 50043.81811800403,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017309094848117173",
            "extra": "mean: 19.98248809956878 usec\nrounds: 18529"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 51414.09040814624,
            "unit": "iter/sec",
            "range": "stddev: 0.000002252182465042932",
            "extra": "mean: 19.449921063692614 usec\nrounds: 18268"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 7208.828381792899,
            "unit": "iter/sec",
            "range": "stddev: 0.00007477647672674826",
            "extra": "mean: 138.71879687490787 usec\nrounds: 3776"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 46848.691925726496,
            "unit": "iter/sec",
            "range": "stddev: 0.000001845653823799312",
            "extra": "mean: 21.34531315378861 usec\nrounds: 26000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 34252.410808182154,
            "unit": "iter/sec",
            "range": "stddev: 0.000001992002241870515",
            "extra": "mean: 29.195025296179207 usec\nrounds: 16366"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 39335.30082140145,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034190089381161924",
            "extra": "mean: 25.42245715980193 usec\nrounds: 11753"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 47342.937651685366,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016623403999917787",
            "extra": "mean: 21.12247464146114 usec\nrounds: 16444"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 39348.46201514033,
            "unit": "iter/sec",
            "range": "stddev: 0.000001951753542951811",
            "extra": "mean: 25.413953907911935 usec\nrounds: 12410"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 24892.677263015295,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029582765041790357",
            "extra": "mean: 40.17245672026474 usec\nrounds: 10282"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 16293.553249323128,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030088330697294697",
            "extra": "mean: 61.37396703457193 usec\nrounds: 910"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 8370.907524277814,
            "unit": "iter/sec",
            "range": "stddev: 0.000008713782367238656",
            "extra": "mean: 119.46136032440201 usec\nrounds: 5434"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2508.17427976985,
            "unit": "iter/sec",
            "range": "stddev: 0.000021340429075034498",
            "extra": "mean: 398.6963777061617 usec\nrounds: 2171"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 895943.3928959938,
            "unit": "iter/sec",
            "range": "stddev: 6.716119040594901e-8",
            "extra": "mean: 1.1161419437088265 usec\nrounds: 173883"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3876.401336008625,
            "unit": "iter/sec",
            "range": "stddev: 0.000009954791344557479",
            "extra": "mean: 257.9712246796561 usec\nrounds: 543"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 4001.5113860792253,
            "unit": "iter/sec",
            "range": "stddev: 0.00008809134687898897",
            "extra": "mean: 249.90557404856554 usec\nrounds: 3599"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1309.568115301158,
            "unit": "iter/sec",
            "range": "stddev: 0.0031228522339702028",
            "extra": "mean: 763.6105280175003 usec\nrounds: 1517"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 622.0439059165728,
            "unit": "iter/sec",
            "range": "stddev: 0.002833741364280329",
            "extra": "mean: 1.607603563813577 msec\nrounds: 619"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 618.3466460983194,
            "unit": "iter/sec",
            "range": "stddev: 0.0029270190700168347",
            "extra": "mean: 1.617215855070711 msec\nrounds: 621"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 9732.117069821436,
            "unit": "iter/sec",
            "range": "stddev: 0.000007032598653331229",
            "extra": "mean: 102.75256584211517 usec\nrounds: 2506"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 12038.7103061819,
            "unit": "iter/sec",
            "range": "stddev: 0.00005286327242651526",
            "extra": "mean: 83.06537615466152 usec\nrounds: 8773"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 183851.58131976528,
            "unit": "iter/sec",
            "range": "stddev: 0.000001415917163328116",
            "extra": "mean: 5.43916996971999 usec\nrounds: 23816"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1584784.9138740285,
            "unit": "iter/sec",
            "range": "stddev: 1.156439908153931e-7",
            "extra": "mean: 631.0004539073294 nsec\nrounds: 74935"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 274710.11400393856,
            "unit": "iter/sec",
            "range": "stddev: 3.746723897310061e-7",
            "extra": "mean: 3.640200884579236 usec\nrounds: 21923"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 4185.851261429344,
            "unit": "iter/sec",
            "range": "stddev: 0.00005269281241785682",
            "extra": "mean: 238.90003192768242 usec\nrounds: 1754"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 3197.8586233970514,
            "unit": "iter/sec",
            "range": "stddev: 0.00005309475400986195",
            "extra": "mean: 312.70925884075217 usec\nrounds: 2743"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1697.2531826754584,
            "unit": "iter/sec",
            "range": "stddev: 0.000054930507597896344",
            "extra": "mean: 589.1872881473428 usec\nrounds: 1527"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 8.836474203230308,
            "unit": "iter/sec",
            "range": "stddev: 0.025145087738238366",
            "extra": "mean: 113.16730824998444 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 2.598788417357666,
            "unit": "iter/sec",
            "range": "stddev: 0.08404241589339778",
            "extra": "mean: 384.7946963749962 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 8.960005728668984,
            "unit": "iter/sec",
            "range": "stddev: 0.026369307540980113",
            "extra": "mean: 111.60707150000349 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 7.303591948776701,
            "unit": "iter/sec",
            "range": "stddev: 0.031489089429751094",
            "extra": "mean: 136.91893071428953 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 2.429258621613133,
            "unit": "iter/sec",
            "range": "stddev: 0.10281131005613875",
            "extra": "mean: 411.64822514284486 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_filesystem",
            "value": 10.840336040129214,
            "unit": "iter/sec",
            "range": "stddev: 0.0035816349170977703",
            "extra": "mean: 92.24806281817811 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_filesystem",
            "value": 9.679462130099148,
            "unit": "iter/sec",
            "range": "stddev: 0.022443202090465637",
            "extra": "mean: 103.31152563636888 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 39740.03511499672,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020245467837329146",
            "extra": "mean: 25.16354092557481 usec\nrounds: 8894"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 30605.54611187382,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019363884930628167",
            "extra": "mean: 32.67381658032355 usec\nrounds: 7720"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 39402.182310842334,
            "unit": "iter/sec",
            "range": "stddev: 0.000002433224721145394",
            "extra": "mean: 25.37930493572761 usec\nrounds: 13534"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 27824.355087591,
            "unit": "iter/sec",
            "range": "stddev: 0.000002170370200547597",
            "extra": "mean: 35.93973685470885 usec\nrounds: 9033"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 34570.90196869582,
            "unit": "iter/sec",
            "range": "stddev: 0.000001597052767956992",
            "extra": "mean: 28.926060445443586 usec\nrounds: 9281"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 26303.33126742627,
            "unit": "iter/sec",
            "range": "stddev: 0.0000058110467516788095",
            "extra": "mean: 38.017998170383386 usec\nrounds: 11480"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 34600.982483380845,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019138526468067645",
            "extra": "mean: 28.90091344892616 usec\nrounds: 8030"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 24878.593784215744,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023526544717286415",
            "extra": "mean: 40.195197874666505 usec\nrounds: 9976"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 28886.17317456216,
            "unit": "iter/sec",
            "range": "stddev: 0.0000038055127813881972",
            "extra": "mean: 34.61863895770809 usec\nrounds: 8096"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 18933.83122429451,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037814698045731583",
            "extra": "mean: 52.81551251586488 usec\nrounds: 7590"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 9941.953920184813,
            "unit": "iter/sec",
            "range": "stddev: 0.000007551760551930853",
            "extra": "mean: 100.58384981746232 usec\nrounds: 3569"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 54211.92503797124,
            "unit": "iter/sec",
            "range": "stddev: 0.000004342906977012878",
            "extra": "mean: 18.44612600086748 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 28876.335445252244,
            "unit": "iter/sec",
            "range": "stddev: 0.000002028580945156401",
            "extra": "mean: 34.63043300268964 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 48199.03811908927,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017114774406743034",
            "extra": "mean: 20.747302000700074 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 24770.91179565101,
            "unit": "iter/sec",
            "range": "stddev: 0.000003259329102361252",
            "extra": "mean: 40.36993100010022 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 783.485903397669,
            "unit": "iter/sec",
            "range": "stddev: 0.000023550854438164166",
            "extra": "mean: 1.2763471501700219 msec\nrounds: 586"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 993.712787981467,
            "unit": "iter/sec",
            "range": "stddev: 0.00010703521778422442",
            "extra": "mean: 1.0063269911533534 msec\nrounds: 904"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6215.008030259436,
            "unit": "iter/sec",
            "range": "stddev: 0.000007098604581195031",
            "extra": "mean: 160.90083796050325 usec\nrounds: 2413"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 6560.59472430605,
            "unit": "iter/sec",
            "range": "stddev: 0.000006552915698117852",
            "extra": "mean: 152.42520564410805 usec\nrounds: 3827"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 6925.587161801006,
            "unit": "iter/sec",
            "range": "stddev: 0.000012790760062213765",
            "extra": "mean: 144.39208931130528 usec\nrounds: 2105"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4551.918859143082,
            "unit": "iter/sec",
            "range": "stddev: 0.000015463554003169566",
            "extra": "mean: 219.68757153730417 usec\nrounds: 2642"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 2066.4837072028695,
            "unit": "iter/sec",
            "range": "stddev: 0.00001925560758882517",
            "extra": "mean: 483.91380803750445 usec\nrounds: 1792"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 2079.1261241996813,
            "unit": "iter/sec",
            "range": "stddev: 0.00001879455582570955",
            "extra": "mean: 480.97130249129566 usec\nrounds: 1967"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 2543.1248049995274,
            "unit": "iter/sec",
            "range": "stddev: 0.000014344963167834931",
            "extra": "mean: 393.2170367864371 usec\nrounds: 2365"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 2535.631923794074,
            "unit": "iter/sec",
            "range": "stddev: 0.00001680344740151128",
            "extra": "mean: 394.37900691189316 usec\nrounds: 2315"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1794.2491566038964,
            "unit": "iter/sec",
            "range": "stddev: 0.00003387981515244487",
            "extra": "mean: 557.3361962130006 usec\nrounds: 1585"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1800.9266098034664,
            "unit": "iter/sec",
            "range": "stddev: 0.00002131833236918517",
            "extra": "mean: 555.2697120229286 usec\nrounds: 1705"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 2211.169752240131,
            "unit": "iter/sec",
            "range": "stddev: 0.000016126565471660637",
            "extra": "mean: 452.24931237726196 usec\nrounds: 2020"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 2208.6036387980234,
            "unit": "iter/sec",
            "range": "stddev: 0.000017992631059868278",
            "extra": "mean: 452.7747679272251 usec\nrounds: 2064"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 597.8628965741786,
            "unit": "iter/sec",
            "range": "stddev: 0.000025582654207169215",
            "extra": "mean: 1.672624285149843 msec\nrounds: 505"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 609.8530886299563,
            "unit": "iter/sec",
            "range": "stddev: 0.000023611602755825834",
            "extra": "mean: 1.6397391743092002 msec\nrounds: 327"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "04e52a1c5108d4353120368659c01db919fa3175",
          "message": "[Feature] Memory-mapped nested tensors (#618)",
          "timestamp": "2024-05-08T20:11:42+01:00",
          "tree_id": "540b10dbf387578eeee273bf0649b21ca34bb1a2",
          "url": "https://github.com/pytorch/tensordict/commit/04e52a1c5108d4353120368659c01db919fa3175"
        },
        "date": 1715195767061,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 60995.1929473271,
            "unit": "iter/sec",
            "range": "stddev: 5.424518139587185e-7",
            "extra": "mean: 16.3947345959469 usec\nrounds: 7709"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 60572.117380684205,
            "unit": "iter/sec",
            "range": "stddev: 8.275021786029377e-7",
            "extra": "mean: 16.509246221577346 usec\nrounds: 17533"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 53838.00997160515,
            "unit": "iter/sec",
            "range": "stddev: 9.407141700357958e-7",
            "extra": "mean: 18.57423780201781 usec\nrounds: 31972"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 53906.71584978153,
            "unit": "iter/sec",
            "range": "stddev: 0.000001241209895358516",
            "extra": "mean: 18.550564326467917 usec\nrounds: 35483"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 392576.3162679687,
            "unit": "iter/sec",
            "range": "stddev: 2.0542671969247108e-7",
            "extra": "mean: 2.547275417698428 usec\nrounds: 112146"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 3807.3751574875646,
            "unit": "iter/sec",
            "range": "stddev: 0.000014155488166027352",
            "extra": "mean: 262.64813910796397 usec\nrounds: 3278"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 3805.137921482005,
            "unit": "iter/sec",
            "range": "stddev: 0.000023882749037721245",
            "extra": "mean: 262.8025634378386 usec\nrounds: 3397"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 13249.570216907909,
            "unit": "iter/sec",
            "range": "stddev: 0.000004044873799254719",
            "extra": "mean: 75.47414622731613 usec\nrounds: 10470"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 3806.812708638336,
            "unit": "iter/sec",
            "range": "stddev: 0.000012274641913161358",
            "extra": "mean: 262.68694483729706 usec\nrounds: 3535"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 12851.393987657695,
            "unit": "iter/sec",
            "range": "stddev: 0.000003918532936677987",
            "extra": "mean: 77.81257044647347 usec\nrounds: 11058"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 3772.049837964427,
            "unit": "iter/sec",
            "range": "stddev: 0.000015050523397688358",
            "extra": "mean: 265.1078439991255 usec\nrounds: 3641"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 259639.8598497242,
            "unit": "iter/sec",
            "range": "stddev: 2.6107575646368703e-7",
            "extra": "mean: 3.851488752839358 usec\nrounds: 106872"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 7168.128055200752,
            "unit": "iter/sec",
            "range": "stddev: 0.00000617599531798641",
            "extra": "mean: 139.5064363107271 usec\nrounds: 6202"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 6998.937830834825,
            "unit": "iter/sec",
            "range": "stddev: 0.000021797120078350703",
            "extra": "mean: 142.87882306860286 usec\nrounds: 6381"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 8532.114185432494,
            "unit": "iter/sec",
            "range": "stddev: 0.00000520661079513867",
            "extra": "mean: 117.20424484090631 usec\nrounds: 7560"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 7167.314106178228,
            "unit": "iter/sec",
            "range": "stddev: 0.000007171083855862275",
            "extra": "mean: 139.52227922284018 usec\nrounds: 6536"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 8454.343893721893,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027018899450740482",
            "extra": "mean: 118.28238980704222 usec\nrounds: 7378"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 6904.853815657056,
            "unit": "iter/sec",
            "range": "stddev: 0.00000617525170910367",
            "extra": "mean: 144.82565839880007 usec\nrounds: 6531"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 639251.9066925496,
            "unit": "iter/sec",
            "range": "stddev: 2.481758632313983e-7",
            "extra": "mean: 1.5643285370457134 usec\nrounds: 160721"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 19983.70078514099,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024939364182243686",
            "extra": "mean: 50.04078127228348 usec\nrounds: 15581"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 20048.08836495686,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017293356038055237",
            "extra": "mean: 49.8800674556061 usec\nrounds: 17997"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 22253.886922262685,
            "unit": "iter/sec",
            "range": "stddev: 0.000002365731046411916",
            "extra": "mean: 44.9359702191892 usec\nrounds: 17629"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 19431.336120538483,
            "unit": "iter/sec",
            "range": "stddev: 0.000004724225697907119",
            "extra": "mean: 51.46326499612256 usec\nrounds: 15804"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 22078.733719771328,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014213509664593077",
            "extra": "mean: 45.29245257867792 usec\nrounds: 18030"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 19633.254440604138,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024300441481011396",
            "extra": "mean: 50.93399074642812 usec\nrounds: 17831"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 741502.4665342616,
            "unit": "iter/sec",
            "range": "stddev: 1.3206723747123206e-7",
            "extra": "mean: 1.348613180848798 usec\nrounds: 144238"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 292372.06617237633,
            "unit": "iter/sec",
            "range": "stddev: 2.6405256050022114e-7",
            "extra": "mean: 3.4202993914282547 usec\nrounds: 113547"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 268764.63536046154,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011034090281425404",
            "extra": "mean: 3.7207276123170776 usec\nrounds: 124767"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 292891.32345119,
            "unit": "iter/sec",
            "range": "stddev: 2.9521656056580444e-7",
            "extra": "mean: 3.414235656477713 usec\nrounds: 78101"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 293982.9481195839,
            "unit": "iter/sec",
            "range": "stddev: 2.4062412703911723e-7",
            "extra": "mean: 3.401557833188435 usec\nrounds: 51372"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 240737.787627546,
            "unit": "iter/sec",
            "range": "stddev: 4.0761520205433373e-7",
            "extra": "mean: 4.153897108779348 usec\nrounds: 102691"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 237766.97398701444,
            "unit": "iter/sec",
            "range": "stddev: 3.0082513716454945e-7",
            "extra": "mean: 4.205798573415897 usec\nrounds: 102062"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 239299.7693505195,
            "unit": "iter/sec",
            "range": "stddev: 4.642173577589677e-7",
            "extra": "mean: 4.17885902152805 usec\nrounds: 75047"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 240861.79778623182,
            "unit": "iter/sec",
            "range": "stddev: 3.25223138838703e-7",
            "extra": "mean: 4.151758432391649 usec\nrounds: 77676"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 96566.40167354456,
            "unit": "iter/sec",
            "range": "stddev: 4.929979899570748e-7",
            "extra": "mean: 10.35556863121639 usec\nrounds: 70988"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 101502.1496833052,
            "unit": "iter/sec",
            "range": "stddev: 6.241841301926283e-7",
            "extra": "mean: 9.852008091652046 usec\nrounds: 75387"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 96562.93830174457,
            "unit": "iter/sec",
            "range": "stddev: 5.885634194391045e-7",
            "extra": "mean: 10.355940048915572 usec\nrounds: 55979"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 101446.92883589871,
            "unit": "iter/sec",
            "range": "stddev: 4.927570655450636e-7",
            "extra": "mean: 9.857370858585648 usec\nrounds: 58405"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 90063.28432786276,
            "unit": "iter/sec",
            "range": "stddev: 0.000001963095704370761",
            "extra": "mean: 11.10330372096625 usec\nrounds: 53345"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 98044.60199318109,
            "unit": "iter/sec",
            "range": "stddev: 0.000001728395987007776",
            "extra": "mean: 10.199439639415836 usec\nrounds: 66318"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 92924.06372843673,
            "unit": "iter/sec",
            "range": "stddev: 8.302134888398047e-7",
            "extra": "mean: 10.761475121475762 usec\nrounds: 51852"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 99666.66351312853,
            "unit": "iter/sec",
            "range": "stddev: 9.153086561344351e-7",
            "extra": "mean: 10.033445133520253 usec\nrounds: 58852"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2841.901358540871,
            "unit": "iter/sec",
            "range": "stddev: 0.00015639037321198403",
            "extra": "mean: 351.87709699869174 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3227.790251512528,
            "unit": "iter/sec",
            "range": "stddev: 0.000015022002387630027",
            "extra": "mean: 309.8094739989392 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2501.2869371397937,
            "unit": "iter/sec",
            "range": "stddev: 0.0015846740683245527",
            "extra": "mean: 399.79419600035726 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3134.2848978392094,
            "unit": "iter/sec",
            "range": "stddev: 0.000019901110069929006",
            "extra": "mean: 319.0520430001129 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 10316.003595671067,
            "unit": "iter/sec",
            "range": "stddev: 0.0000058426326544367",
            "extra": "mean: 96.93676342063634 usec\nrounds: 7731"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 2482.221144299163,
            "unit": "iter/sec",
            "range": "stddev: 0.000008192049091604643",
            "extra": "mean: 402.8649914197482 usec\nrounds: 2331"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1397.212305987929,
            "unit": "iter/sec",
            "range": "stddev: 0.00014004726281393904",
            "extra": "mean: 715.7108448833252 usec\nrounds: 909"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 515884.7699755079,
            "unit": "iter/sec",
            "range": "stddev: 2.1971340241928237e-7",
            "extra": "mean: 1.9384173718628597 usec\nrounds: 122325"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 103605.27038736753,
            "unit": "iter/sec",
            "range": "stddev: 5.334297160915962e-7",
            "extra": "mean: 9.652018630530293 usec\nrounds: 25872"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 80471.41187128329,
            "unit": "iter/sec",
            "range": "stddev: 7.413555950271211e-7",
            "extra": "mean: 12.426773393755456 usec\nrounds: 26054"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 62422.102853134114,
            "unit": "iter/sec",
            "range": "stddev: 9.407246859840364e-7",
            "extra": "mean: 16.019966555000344 usec\nrounds: 22993"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 74718.56565055964,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014495946124234337",
            "extra": "mean: 13.383554559609108 usec\nrounds: 14168"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 85594.87054459106,
            "unit": "iter/sec",
            "range": "stddev: 9.564151439528643e-7",
            "extra": "mean: 11.682943074013357 usec\nrounds: 11594"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 42536.15197635514,
            "unit": "iter/sec",
            "range": "stddev: 0.000004061067345934779",
            "extra": "mean: 23.50941384062848 usec\nrounds: 8641"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 16564.738312643673,
            "unit": "iter/sec",
            "range": "stddev: 0.000010879263736289083",
            "extra": "mean: 60.36919999132806 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 52235.57834176185,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016775451979954801",
            "extra": "mean: 19.144039977068836 usec\nrounds: 15759"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 24087.789990547597,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035568535527369393",
            "extra": "mean: 41.51480897136743 usec\nrounds: 7758"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 29203.17672185611,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026611516297345917",
            "extra": "mean: 34.24284999965721 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 16461.15978686479,
            "unit": "iter/sec",
            "range": "stddev: 0.000003482426545114131",
            "extra": "mean: 60.749060998603 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 11738.069027585962,
            "unit": "iter/sec",
            "range": "stddev: 0.000005534034416466707",
            "extra": "mean: 85.19288799971036 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 20516.288291822362,
            "unit": "iter/sec",
            "range": "stddev: 0.00000250083882041762",
            "extra": "mean: 48.74175999947283 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 51425.84333648516,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017936967132523111",
            "extra": "mean: 19.445475953731783 usec\nrounds: 16468"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 52354.6399690221,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014244231335188814",
            "extra": "mean: 19.100503806189735 usec\nrounds: 19442"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 6866.991003304397,
            "unit": "iter/sec",
            "range": "stddev: 0.00009283838605329946",
            "extra": "mean: 145.6241896223251 usec\nrounds: 3623"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 48631.85281613894,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021563822996716883",
            "extra": "mean: 20.56265476416602 usec\nrounds: 25933"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 34435.17556023319,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023604093511716292",
            "extra": "mean: 29.040072650445 usec\nrounds: 12278"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 39450.098934605376,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035666110252915305",
            "extra": "mean: 25.348478888675395 usec\nrounds: 11013"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 47643.834717794154,
            "unit": "iter/sec",
            "range": "stddev: 0.000001794691466935001",
            "extra": "mean: 20.989074576453376 usec\nrounds: 15930"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 40252.63203423943,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020836285717568636",
            "extra": "mean: 24.843095953312726 usec\nrounds: 14705"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 25030.193454255124,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030961717183519204",
            "extra": "mean: 39.951748748070514 usec\nrounds: 10786"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 16617.12073088463,
            "unit": "iter/sec",
            "range": "stddev: 0.000004223069544758307",
            "extra": "mean: 60.17889718652624 usec\nrounds: 1031"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 8299.086955627294,
            "unit": "iter/sec",
            "range": "stddev: 0.000004795349708047813",
            "extra": "mean: 120.49518282513453 usec\nrounds: 5508"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2537.766156996103,
            "unit": "iter/sec",
            "range": "stddev: 0.000007284471695152258",
            "extra": "mean: 394.0473385395279 usec\nrounds: 2177"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 901456.8767384468,
            "unit": "iter/sec",
            "range": "stddev: 5.856744498810153e-8",
            "extra": "mean: 1.1093154046571028 usec\nrounds: 176648"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3822.0447681398973,
            "unit": "iter/sec",
            "range": "stddev: 0.00005149789003789951",
            "extra": "mean: 261.64005412387604 usec\nrounds: 776"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 3884.126764878456,
            "unit": "iter/sec",
            "range": "stddev: 0.000009898771994216828",
            "extra": "mean: 257.45812650666477 usec\nrounds: 3486"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1297.1095503159602,
            "unit": "iter/sec",
            "range": "stddev: 0.0028150067713337487",
            "extra": "mean: 770.9449057378476 usec\nrounds: 1464"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 618.1019266101094,
            "unit": "iter/sec",
            "range": "stddev: 0.00251465922940355",
            "extra": "mean: 1.6178561446723767 msec\nrounds: 629"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 616.1701097154752,
            "unit": "iter/sec",
            "range": "stddev: 0.002532893680660726",
            "extra": "mean: 1.622928448219865 msec\nrounds: 618"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 9044.751218187681,
            "unit": "iter/sec",
            "range": "stddev: 0.0000738954443409797",
            "extra": "mean: 110.56136049260759 usec\nrounds: 2516"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 11551.384027825996,
            "unit": "iter/sec",
            "range": "stddev: 0.0000077837102353575",
            "extra": "mean: 86.56971299639173 usec\nrounds: 8756"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 177316.69790046074,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016130546984225837",
            "extra": "mean: 5.639626791163032 usec\nrounds: 22888"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1579475.2872555174,
            "unit": "iter/sec",
            "range": "stddev: 9.352466829896999e-8",
            "extra": "mean: 633.1216499990901 nsec\nrounds: 76605"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 270760.06199444295,
            "unit": "iter/sec",
            "range": "stddev: 3.8149396511165937e-7",
            "extra": "mean: 3.693306880763396 usec\nrounds: 29575"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 4065.219985928392,
            "unit": "iter/sec",
            "range": "stddev: 0.00005260111975140051",
            "extra": "mean: 245.9891478103185 usec\nrounds: 1759"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 3135.6329220743837,
            "unit": "iter/sec",
            "range": "stddev: 0.00005091916097821194",
            "extra": "mean: 318.91488093524936 usec\nrounds: 2780"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1693.3545684208382,
            "unit": "iter/sec",
            "range": "stddev: 0.00005878175579547083",
            "extra": "mean: 590.5437754436534 usec\nrounds: 1238"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 8.970833988652743,
            "unit": "iter/sec",
            "range": "stddev: 0.02231271573007165",
            "extra": "mean: 111.47235600000016 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 2.607255226130106,
            "unit": "iter/sec",
            "range": "stddev: 0.07985224890149585",
            "extra": "mean: 383.54511287499804 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.032482549388646,
            "unit": "iter/sec",
            "range": "stddev: 0.02074234278229296",
            "extra": "mean: 110.71153412498802 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 7.910095338955159,
            "unit": "iter/sec",
            "range": "stddev: 0.0036060628137294134",
            "extra": "mean: 126.42072657143086 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 1.5416242758290686,
            "unit": "iter/sec",
            "range": "stddev: 0.23577769788377562",
            "extra": "mean: 648.6664848749939 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_filesystem",
            "value": 10.907259080540687,
            "unit": "iter/sec",
            "range": "stddev: 0.0024754173912755603",
            "extra": "mean: 91.68206169999848 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_filesystem",
            "value": 9.765578929077027,
            "unit": "iter/sec",
            "range": "stddev: 0.01999818192824921",
            "extra": "mean: 102.40048309092035 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 38781.268206507266,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031264134036415113",
            "extra": "mean: 25.785644622942115 usec\nrounds: 7969"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 29998.233883135505,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017317488292037918",
            "extra": "mean: 33.335295800936564 usec\nrounds: 7931"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 39460.46009937513,
            "unit": "iter/sec",
            "range": "stddev: 0.000001359132609873761",
            "extra": "mean: 25.341823118170772 usec\nrounds: 15304"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 27224.633097292244,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023498348439220693",
            "extra": "mean: 36.731440839856894 usec\nrounds: 10049"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 34288.97664549211,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016285063501292952",
            "extra": "mean: 29.16389165937583 usec\nrounds: 11270"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 25495.1581381551,
            "unit": "iter/sec",
            "range": "stddev: 0.000005623650674097305",
            "extra": "mean: 39.223133842948684 usec\nrounds: 9683"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 33683.593080203435,
            "unit": "iter/sec",
            "range": "stddev: 0.000003426957457003507",
            "extra": "mean: 29.688044194659305 usec\nrounds: 8010"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 24089.428595516798,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023124608870296173",
            "extra": "mean: 41.51198506161772 usec\nrounds: 10242"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 28693.504164339047,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023529381966610525",
            "extra": "mean: 34.85109362288428 usec\nrounds: 8342"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 19179.64073046011,
            "unit": "iter/sec",
            "range": "stddev: 0.00000544920088064838",
            "extra": "mean: 52.13862001136716 usec\nrounds: 7016"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 9919.333269932826,
            "unit": "iter/sec",
            "range": "stddev: 0.000007397835351768515",
            "extra": "mean: 100.81322733970123 usec\nrounds: 4060"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 57058.02978205492,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019430014844810484",
            "extra": "mean: 17.52601700093237 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 28839.455051087338,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026180179827288794",
            "extra": "mean: 34.674718999667675 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 49687.06588818966,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017172742122587695",
            "extra": "mean: 20.125962000861364 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 24832.196432832592,
            "unit": "iter/sec",
            "range": "stddev: 0.000004277535403983323",
            "extra": "mean: 40.27029999963361 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 762.9999056292864,
            "unit": "iter/sec",
            "range": "stddev: 0.0000591932888639838",
            "extra": "mean: 1.3106161516170138 msec\nrounds: 587"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 984.6101504091791,
            "unit": "iter/sec",
            "range": "stddev: 0.00009243716895744124",
            "extra": "mean: 1.0156303990817335 msec\nrounds: 872"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6129.900216808726,
            "unit": "iter/sec",
            "range": "stddev: 0.000012597904527932815",
            "extra": "mean: 163.13479251389964 usec\nrounds: 2458"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 6522.367925312128,
            "unit": "iter/sec",
            "range": "stddev: 0.000006351265065724925",
            "extra": "mean: 153.3185510923389 usec\nrounds: 3983"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 6666.901151616955,
            "unit": "iter/sec",
            "range": "stddev: 0.000008552475511000096",
            "extra": "mean: 149.99472427418027 usec\nrounds: 2031"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4484.84049165429,
            "unit": "iter/sec",
            "range": "stddev: 0.000013361616578636006",
            "extra": "mean: 222.97337037089082 usec\nrounds: 2700"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 2064.9192700328035,
            "unit": "iter/sec",
            "range": "stddev: 0.000020708961460000342",
            "extra": "mean: 484.2804338709638 usec\nrounds: 1860"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 2059.3202518063013,
            "unit": "iter/sec",
            "range": "stddev: 0.000022937789802487233",
            "extra": "mean: 485.5971280440064 usec\nrounds: 1601"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 2537.314588797099,
            "unit": "iter/sec",
            "range": "stddev: 0.000016787433323892787",
            "extra": "mean: 394.11746750491994 usec\nrounds: 2385"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 2524.5165016892906,
            "unit": "iter/sec",
            "range": "stddev: 0.000021165699411183556",
            "extra": "mean: 396.1154539219078 usec\nrounds: 2333"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1805.345640563796,
            "unit": "iter/sec",
            "range": "stddev: 0.000025302734293194936",
            "extra": "mean: 553.9105518252491 usec\nrounds: 1698"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1817.1873660909935,
            "unit": "iter/sec",
            "range": "stddev: 0.000023110032196670352",
            "extra": "mean: 550.3009863815695 usec\nrounds: 1689"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 2194.1357253261162,
            "unit": "iter/sec",
            "range": "stddev: 0.000020809536282359652",
            "extra": "mean: 455.7603198641547 usec\nrounds: 2054"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 2195.0949109656835,
            "unit": "iter/sec",
            "range": "stddev: 0.00001302975017117122",
            "extra": "mean: 455.5611673119282 usec\nrounds: 2068"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 595.5464509082151,
            "unit": "iter/sec",
            "range": "stddev: 0.00008381464411770917",
            "extra": "mean: 1.6791301475728528 msec\nrounds: 515"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 611.5654457011526,
            "unit": "iter/sec",
            "range": "stddev: 0.00002607780562099183",
            "extra": "mean: 1.6351479748067057 msec\nrounds: 516"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5c4cef710fa328350539b55497e339faf5b69378",
          "message": "[BugFix] Ensure dtype is preserved with autocast (#773)",
          "timestamp": "2024-05-13T09:36:40+01:00",
          "tree_id": "00641008a9d0eed7521a35fc6041833fdcc432dd",
          "url": "https://github.com/pytorch/tensordict/commit/5c4cef710fa328350539b55497e339faf5b69378"
        },
        "date": 1715589666927,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 60402.907483535455,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010239560823170916",
            "extra": "mean: 16.555494456497456 usec\nrounds: 7847"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 60712.97745864826,
            "unit": "iter/sec",
            "range": "stddev: 0.000001089176284908949",
            "extra": "mean: 16.470943146893134 usec\nrounds: 17853"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 53524.517910835275,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011409329305320942",
            "extra": "mean: 18.68302675170035 usec\nrounds: 31587"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 53916.635910674086,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010847403118334846",
            "extra": "mean: 18.54715122910749 usec\nrounds: 34742"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 387871.5022975431,
            "unit": "iter/sec",
            "range": "stddev: 2.2936121558273024e-7",
            "extra": "mean: 2.5781734261902085 usec\nrounds: 105186"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 3788.2938266614724,
            "unit": "iter/sec",
            "range": "stddev: 0.0000056327983157419446",
            "extra": "mean: 263.971076626143 usec\nrounds: 3106"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 3782.8560651604603,
            "unit": "iter/sec",
            "range": "stddev: 0.000020368055911105463",
            "extra": "mean: 264.35052848292344 usec\nrounds: 3546"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 13043.553391111996,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021419865442616115",
            "extra": "mean: 76.66622507034086 usec\nrounds: 9637"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 3747.4489385073803,
            "unit": "iter/sec",
            "range": "stddev: 0.000016362886349996017",
            "extra": "mean: 266.8481989772762 usec\nrounds: 3518"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 12931.389133479237,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024509821917102767",
            "extra": "mean: 77.33121242257027 usec\nrounds: 10964"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 3712.238965615328,
            "unit": "iter/sec",
            "range": "stddev: 0.000015347501423571166",
            "extra": "mean: 269.3792100299888 usec\nrounds: 3709"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 246581.46404019746,
            "unit": "iter/sec",
            "range": "stddev: 2.4986629961986996e-7",
            "extra": "mean: 4.055454873270527 usec\nrounds: 107331"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 7178.179422849269,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031334833087253556",
            "extra": "mean: 139.3110900539548 usec\nrounds: 6063"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 6968.010351455381,
            "unit": "iter/sec",
            "range": "stddev: 0.000023862663376712645",
            "extra": "mean: 143.5129900160286 usec\nrounds: 5709"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 8449.485617883229,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035833527313096355",
            "extra": "mean: 118.35039968391837 usec\nrounds: 7596"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 7242.290396673185,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034014993459336857",
            "extra": "mean: 138.07786559613234 usec\nrounds: 6540"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 8465.936970221448,
            "unit": "iter/sec",
            "range": "stddev: 0.000002972353569183009",
            "extra": "mean: 118.12041638361531 usec\nrounds: 7349"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 6990.975813615082,
            "unit": "iter/sec",
            "range": "stddev: 0.000008902629973306841",
            "extra": "mean: 143.04154765526118 usec\nrounds: 6568"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 846868.6758891774,
            "unit": "iter/sec",
            "range": "stddev: 2.4232200307276053e-7",
            "extra": "mean: 1.1808206259961629 usec\nrounds: 197668"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 19840.253680560214,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026346164550034654",
            "extra": "mean: 50.40258134299036 usec\nrounds: 15158"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 19916.31388975838,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015533288422854938",
            "extra": "mean: 50.21009437465397 usec\nrounds: 17812"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 21825.466480363186,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027432667423625728",
            "extra": "mean: 45.81803559157465 usec\nrounds: 17448"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 19183.51212082081,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014803589625111565",
            "extra": "mean: 52.12809800946985 usec\nrounds: 15672"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 21797.72657244435,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037777547778612523",
            "extra": "mean: 45.876343878180045 usec\nrounds: 18050"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 19457.644890038384,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023815672731485158",
            "extra": "mean: 51.39368128318367 usec\nrounds: 17583"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 744339.1975847101,
            "unit": "iter/sec",
            "range": "stddev: 1.510074449007742e-7",
            "extra": "mean: 1.3434735175104011 usec\nrounds: 152138"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 291308.5076524954,
            "unit": "iter/sec",
            "range": "stddev: 2.7908059824358225e-7",
            "extra": "mean: 3.43278680069622 usec\nrounds: 112158"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 288422.7365587876,
            "unit": "iter/sec",
            "range": "stddev: 5.849978559520305e-7",
            "extra": "mean: 3.4671330420449555 usec\nrounds: 119962"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 292948.74193140323,
            "unit": "iter/sec",
            "range": "stddev: 4.2968652683445533e-7",
            "extra": "mean: 3.413566460149399 usec\nrounds: 76023"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 291425.7772515777,
            "unit": "iter/sec",
            "range": "stddev: 4.827290223181588e-7",
            "extra": "mean: 3.4314054488623182 usec\nrounds: 65885"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 239466.96399323127,
            "unit": "iter/sec",
            "range": "stddev: 5.548176742683807e-7",
            "extra": "mean: 4.175941362952536 usec\nrounds: 93985"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 241198.85213403267,
            "unit": "iter/sec",
            "range": "stddev: 3.274434495468744e-7",
            "extra": "mean: 4.1459567122828025 usec\nrounds: 100121"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 188465.67079297293,
            "unit": "iter/sec",
            "range": "stddev: 5.211533425625787e-7",
            "extra": "mean: 5.306006106005835 usec\nrounds: 58303"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 186226.5394615214,
            "unit": "iter/sec",
            "range": "stddev: 3.408130676074026e-7",
            "extra": "mean: 5.369803911362604 usec\nrounds: 63456"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 95123.14616211077,
            "unit": "iter/sec",
            "range": "stddev: 4.93251579085495e-7",
            "extra": "mean: 10.512688450146296 usec\nrounds: 70191"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 99041.27023476949,
            "unit": "iter/sec",
            "range": "stddev: 5.113457843635949e-7",
            "extra": "mean: 10.096801036876641 usec\nrounds: 72913"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 95458.29329282598,
            "unit": "iter/sec",
            "range": "stddev: 5.411361552043427e-7",
            "extra": "mean: 10.475779164963903 usec\nrounds: 57432"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 100720.80995042867,
            "unit": "iter/sec",
            "range": "stddev: 5.602034134192754e-7",
            "extra": "mean: 9.928434853653041 usec\nrounds: 60057"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 89881.98860076412,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010439529881967",
            "extra": "mean: 11.12569954856894 usec\nrounds: 53150"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 97407.87676510836,
            "unit": "iter/sec",
            "range": "stddev: 5.743731227523022e-7",
            "extra": "mean: 10.266110228553933 usec\nrounds: 68367"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 90834.57662297088,
            "unit": "iter/sec",
            "range": "stddev: 6.055972462081128e-7",
            "extra": "mean: 11.009023624899168 usec\nrounds: 52233"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 98535.5377732702,
            "unit": "iter/sec",
            "range": "stddev: 6.34914410518569e-7",
            "extra": "mean: 10.148622746658116 usec\nrounds: 58303"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2858.762893569651,
            "unit": "iter/sec",
            "range": "stddev: 0.00014724536686082444",
            "extra": "mean: 349.8016580001604 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3252.3957521086168,
            "unit": "iter/sec",
            "range": "stddev: 0.000029423379460831805",
            "extra": "mean: 307.4656580004671 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2501.6645826003714,
            "unit": "iter/sec",
            "range": "stddev: 0.001716902761795156",
            "extra": "mean: 399.7338439993996 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3211.6204856511363,
            "unit": "iter/sec",
            "range": "stddev: 0.000016287637611091836",
            "extra": "mean: 311.36929299952953 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 10401.793738321934,
            "unit": "iter/sec",
            "range": "stddev: 0.0000046889623607322085",
            "extra": "mean: 96.13726489459545 usec\nrounds: 7637"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 2446.4633859310006,
            "unit": "iter/sec",
            "range": "stddev: 0.000022369205552305292",
            "extra": "mean: 408.75330722329625 usec\nrounds: 2298"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1479.0932723352096,
            "unit": "iter/sec",
            "range": "stddev: 0.00006280762659986554",
            "extra": "mean: 676.0898847313316 usec\nrounds: 989"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 516543.5436961698,
            "unit": "iter/sec",
            "range": "stddev: 1.8242204524478121e-7",
            "extra": "mean: 1.9359452115971054 usec\nrounds: 125866"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 99985.4257110945,
            "unit": "iter/sec",
            "range": "stddev: 8.560993682601467e-7",
            "extra": "mean: 10.00145764133141 usec\nrounds: 22522"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 78427.54337280197,
            "unit": "iter/sec",
            "range": "stddev: 7.809563484228909e-7",
            "extra": "mean: 12.750622510851104 usec\nrounds: 26414"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 62227.477458596906,
            "unit": "iter/sec",
            "range": "stddev: 8.701578798070889e-7",
            "extra": "mean: 16.070071306768792 usec\nrounds: 23448"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 75483.46009056081,
            "unit": "iter/sec",
            "range": "stddev: 0.00001088634070078648",
            "extra": "mean: 13.247935359617276 usec\nrounds: 15300"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 85545.01315856213,
            "unit": "iter/sec",
            "range": "stddev: 8.892439399462868e-7",
            "extra": "mean: 11.689752132557956 usec\nrounds: 12192"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 44196.401135201304,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016103985650732417",
            "extra": "mean: 22.626276672186453 usec\nrounds: 11454"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 17005.18318071279,
            "unit": "iter/sec",
            "range": "stddev: 0.000012007868201204504",
            "extra": "mean: 58.805599996958335 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 52443.262463818435,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012520292499033155",
            "extra": "mean: 19.06822636539667 usec\nrounds: 16628"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 25206.46857880529,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035699211982247576",
            "extra": "mean: 39.67235619990197 usec\nrounds: 7347"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 28185.105795140615,
            "unit": "iter/sec",
            "range": "stddev: 0.000002377914428035201",
            "extra": "mean: 35.47973199988519 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 15929.288994177665,
            "unit": "iter/sec",
            "range": "stddev: 0.000003522985341422094",
            "extra": "mean: 62.7774409997528 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 11713.444816040834,
            "unit": "iter/sec",
            "range": "stddev: 0.000004918181236381019",
            "extra": "mean: 85.37198200059493 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 19081.856623222116,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032328214572392936",
            "extra": "mean: 52.405802000578205 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 50834.49166047559,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023967659794957608",
            "extra": "mean: 19.67168289355614 usec\nrounds: 16947"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 52843.67109548078,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020340488839249955",
            "extra": "mean: 18.923742035127468 usec\nrounds: 19053"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 6947.636924735142,
            "unit": "iter/sec",
            "range": "stddev: 0.00010732168189397801",
            "extra": "mean: 143.93383114764907 usec\nrounds: 3737"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 47605.69424688015,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018255443654120667",
            "extra": "mean: 21.005890488941567 usec\nrounds: 25276"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 33642.18362972654,
            "unit": "iter/sec",
            "range": "stddev: 0.000004124488811593921",
            "extra": "mean: 29.724586578749626 usec\nrounds: 16228"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 39673.915995693475,
            "unit": "iter/sec",
            "range": "stddev: 0.000003418162836288725",
            "extra": "mean: 25.205477576464798 usec\nrounds: 11372"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 46393.52180763828,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032758962964263756",
            "extra": "mean: 21.554733528234944 usec\nrounds: 18471"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 39853.07376427304,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031156130274558026",
            "extra": "mean: 25.092167442714715 usec\nrounds: 14602"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 24460.12958520117,
            "unit": "iter/sec",
            "range": "stddev: 0.00005355271047472886",
            "extra": "mean: 40.88285781629784 usec\nrounds: 10817"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 16693.903022596794,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031782307010058947",
            "extra": "mean: 59.90210909015133 usec\nrounds: 935"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 8293.22453262195,
            "unit": "iter/sec",
            "range": "stddev: 0.000012444342139313462",
            "extra": "mean: 120.58036003564517 usec\nrounds: 5630"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2553.9371932895633,
            "unit": "iter/sec",
            "range": "stddev: 0.000006669867835434381",
            "extra": "mean: 391.55230701345624 usec\nrounds: 2267"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 925978.6946271268,
            "unit": "iter/sec",
            "range": "stddev: 1.5036522617659434e-7",
            "extra": "mean: 1.079938454094414 usec\nrounds: 180181"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3829.0291036459507,
            "unit": "iter/sec",
            "range": "stddev: 0.00002603279853574599",
            "extra": "mean: 261.1628099268854 usec\nrounds: 826"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 3924.921947896387,
            "unit": "iter/sec",
            "range": "stddev: 0.000012974986208382375",
            "extra": "mean: 254.7821366322872 usec\nrounds: 3557"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1305.1283555982013,
            "unit": "iter/sec",
            "range": "stddev: 0.0030721428840790616",
            "extra": "mean: 766.2081631363019 usec\nrounds: 1416"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 624.1354463502921,
            "unit": "iter/sec",
            "range": "stddev: 0.0027454586067042924",
            "extra": "mean: 1.6022163231516837 msec\nrounds: 622"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 623.1335351261514,
            "unit": "iter/sec",
            "range": "stddev: 0.0029688963295096222",
            "extra": "mean: 1.6047924620162728 msec\nrounds: 645"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 9390.802684687245,
            "unit": "iter/sec",
            "range": "stddev: 0.00000893938529076933",
            "extra": "mean: 106.48716979546508 usec\nrounds: 2503"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 11737.89131704996,
            "unit": "iter/sec",
            "range": "stddev: 0.000048695659861100115",
            "extra": "mean: 85.19417781177124 usec\nrounds: 8464"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 189355.29022019252,
            "unit": "iter/sec",
            "range": "stddev: 0.000001730604426281885",
            "extra": "mean: 5.281077697048476 usec\nrounds: 22562"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1601528.3163122337,
            "unit": "iter/sec",
            "range": "stddev: 1.31423105715171e-7",
            "extra": "mean: 624.4035711479984 nsec\nrounds: 68885"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 289768.56191441626,
            "unit": "iter/sec",
            "range": "stddev: 4.043221414419865e-7",
            "extra": "mean: 3.4510299992286675 usec\nrounds: 26434"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 3968.611267523831,
            "unit": "iter/sec",
            "range": "stddev: 0.00005404738113930938",
            "extra": "mean: 251.97731210997102 usec\nrounds: 1602"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 3064.176485122401,
            "unit": "iter/sec",
            "range": "stddev: 0.000054151107242220465",
            "extra": "mean: 326.351959443372 usec\nrounds: 2441"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1681.8708799316996,
            "unit": "iter/sec",
            "range": "stddev: 0.0000644903570192652",
            "extra": "mean: 594.5759641433413 usec\nrounds: 1506"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 8.492303368436547,
            "unit": "iter/sec",
            "range": "stddev: 0.03074019734626737",
            "extra": "mean: 117.75368314286945 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 2.604197632204143,
            "unit": "iter/sec",
            "range": "stddev: 0.08342419089673161",
            "extra": "mean: 383.99543399999914 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 8.777975385608265,
            "unit": "iter/sec",
            "range": "stddev: 0.02678716665485085",
            "extra": "mean: 113.92148600000951 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 7.844969628849952,
            "unit": "iter/sec",
            "range": "stddev: 0.003922471223328542",
            "extra": "mean: 127.47021942857373 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 1.526030205289614,
            "unit": "iter/sec",
            "range": "stddev: 0.24300475012218795",
            "extra": "mean: 655.2950240000115 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_filesystem",
            "value": 11.05036499205635,
            "unit": "iter/sec",
            "range": "stddev: 0.002973293835848382",
            "extra": "mean: 90.49474842856853 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_filesystem",
            "value": 10.75044085701152,
            "unit": "iter/sec",
            "range": "stddev: 0.0034439102503247858",
            "extra": "mean: 93.01944109090114 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 39291.503282333746,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017094146435190791",
            "extra": "mean: 25.450795120114943 usec\nrounds: 9303"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 30534.4114725691,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018118570834150386",
            "extra": "mean: 32.74993529508045 usec\nrounds: 7882"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 39705.87650353437,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015504054271605192",
            "extra": "mean: 25.18518889542676 usec\nrounds: 14553"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 26921.402374255642,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036779611845735785",
            "extra": "mean: 37.14516748043849 usec\nrounds: 9416"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 34187.707974838435,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015805545577690144",
            "extra": "mean: 29.25027909844038 usec\nrounds: 10516"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 25429.480400382843,
            "unit": "iter/sec",
            "range": "stddev: 0.000007724835536665473",
            "extra": "mean: 39.324437002061 usec\nrounds: 11929"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 34381.34711543341,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034200641200347507",
            "extra": "mean: 29.08553863938365 usec\nrounds: 8217"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 24689.269555100946,
            "unit": "iter/sec",
            "range": "stddev: 0.000002844677388297488",
            "extra": "mean: 40.503425902018805 usec\nrounds: 10810"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 28590.815586238677,
            "unit": "iter/sec",
            "range": "stddev: 0.000002184114071492099",
            "extra": "mean: 34.976267010771096 usec\nrounds: 7951"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 17741.542135092473,
            "unit": "iter/sec",
            "range": "stddev: 0.000005611358584990631",
            "extra": "mean: 56.36488600514702 usec\nrounds: 7474"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 9862.04532906797,
            "unit": "iter/sec",
            "range": "stddev: 0.000007297690412242094",
            "extra": "mean: 101.3988444215057 usec\nrounds: 3773"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 56914.26344954447,
            "unit": "iter/sec",
            "range": "stddev: 0.000002949485388344889",
            "extra": "mean: 17.570287997955347 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 28957.901553264106,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023538895939648682",
            "extra": "mean: 34.532888999592615 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 49243.63500217934,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024237556812102154",
            "extra": "mean: 20.307192999780455 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 24932.07754102291,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030604470060788503",
            "extra": "mean: 40.10897200021191 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 751.9077028751278,
            "unit": "iter/sec",
            "range": "stddev: 0.00003805701271602924",
            "extra": "mean: 1.3299504662290629 msec\nrounds: 607"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 972.0907742201056,
            "unit": "iter/sec",
            "range": "stddev: 0.00010391379750029001",
            "extra": "mean: 1.0287105139973018 msec\nrounds: 893"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6113.170701481291,
            "unit": "iter/sec",
            "range": "stddev: 0.000011093749518061113",
            "extra": "mean: 163.58123285477512 usec\nrounds: 2362"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 6596.560727480478,
            "unit": "iter/sec",
            "range": "stddev: 0.0000097580823077852",
            "extra": "mean: 151.59414751297905 usec\nrounds: 3898"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 6775.5410004538135,
            "unit": "iter/sec",
            "range": "stddev: 0.00001314649847631119",
            "extra": "mean: 147.58969061408115 usec\nrounds: 1907"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4477.2663826778435,
            "unit": "iter/sec",
            "range": "stddev: 0.00001542821780484852",
            "extra": "mean: 223.3505703098019 usec\nrounds: 2553"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 2058.665666754777,
            "unit": "iter/sec",
            "range": "stddev: 0.000027670437374159105",
            "extra": "mean: 485.75153127043313 usec\nrounds: 1455"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 2066.9633374672794,
            "unit": "iter/sec",
            "range": "stddev: 0.000025915667218188307",
            "extra": "mean: 483.8015178466271 usec\nrounds: 1933"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 2538.2060684729126,
            "unit": "iter/sec",
            "range": "stddev: 0.000018144718645574372",
            "extra": "mean: 393.97904386921607 usec\nrounds: 2348"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 2504.4719607356883,
            "unit": "iter/sec",
            "range": "stddev: 0.000038078887767010164",
            "extra": "mean: 399.28576389661396 usec\nrounds: 2177"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1809.3884285825086,
            "unit": "iter/sec",
            "range": "stddev: 0.00002930580694611907",
            "extra": "mean: 552.6729276053838 usec\nrounds: 1699"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1816.2640154810808,
            "unit": "iter/sec",
            "range": "stddev: 0.000021240158615326697",
            "extra": "mean: 550.5807478848972 usec\nrounds: 1654"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 2179.9305897221466,
            "unit": "iter/sec",
            "range": "stddev: 0.000018786404926938025",
            "extra": "mean: 458.730202105866 usec\nrounds: 1900"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 2065.91081631818,
            "unit": "iter/sec",
            "range": "stddev: 0.00007684857888206863",
            "extra": "mean: 484.0480005725405 usec\nrounds: 1753"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 542.1342487157466,
            "unit": "iter/sec",
            "range": "stddev: 0.0035263131636871243",
            "extra": "mean: 1.8445615682257384 msec\nrounds: 491"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 608.0649149254926,
            "unit": "iter/sec",
            "range": "stddev: 0.000021503631166604747",
            "extra": "mean: 1.644561255639181 msec\nrounds: 532"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e4e5c1eb1aaa387949ba4eef270353464b6d5756",
          "message": "[Feature] TensorDict.replace (#774)",
          "timestamp": "2024-05-13T11:09:27+01:00",
          "tree_id": "20dd8e632b99b77da82f980ef44297a4051ddd0a",
          "url": "https://github.com/pytorch/tensordict/commit/e4e5c1eb1aaa387949ba4eef270353464b6d5756"
        },
        "date": 1715595299043,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 56516.71651313352,
            "unit": "iter/sec",
            "range": "stddev: 8.647624608165918e-7",
            "extra": "mean: 17.69388000039983 usec\nrounds: 7600"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 55786.554814173716,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012880864471654044",
            "extra": "mean: 17.92546615095739 usec\nrounds: 11374"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 49841.350367678715,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015613351265282637",
            "extra": "mean: 20.06366185151523 usec\nrounds: 31359"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 50157.98547429887,
            "unit": "iter/sec",
            "range": "stddev: 9.732065768919488e-7",
            "extra": "mean: 19.937004856632523 usec\nrounds: 33150"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 376389.9757114133,
            "unit": "iter/sec",
            "range": "stddev: 2.4891423997584426e-7",
            "extra": "mean: 2.6568188967038875 usec\nrounds: 111533"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 3719.5789422833454,
            "unit": "iter/sec",
            "range": "stddev: 0.000004799072965621193",
            "extra": "mean: 268.84763450836397 usec\nrounds: 3234"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 3576.3724445642542,
            "unit": "iter/sec",
            "range": "stddev: 0.0000529073068707957",
            "extra": "mean: 279.6129361526384 usec\nrounds: 3571"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 13027.207525956801,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023187621436551268",
            "extra": "mean: 76.762421878019 usec\nrounds: 10202"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 3656.0950033121176,
            "unit": "iter/sec",
            "range": "stddev: 0.000017672345472057154",
            "extra": "mean: 273.5158684591301 usec\nrounds: 3421"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 12666.324486620008,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026223327079034627",
            "extra": "mean: 78.94950117978927 usec\nrounds: 11020"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 3663.2338910018193,
            "unit": "iter/sec",
            "range": "stddev: 0.000005816413625970679",
            "extra": "mean: 272.98284241591807 usec\nrounds: 3560"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 258868.14881394853,
            "unit": "iter/sec",
            "range": "stddev: 2.611763452100095e-7",
            "extra": "mean: 3.8629704140184176 usec\nrounds: 109207"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 7130.888497740876,
            "unit": "iter/sec",
            "range": "stddev: 0.000003109443301881203",
            "extra": "mean: 140.23497917781327 usec\nrounds: 5619"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 6935.010827200775,
            "unit": "iter/sec",
            "range": "stddev: 0.000022108284343197854",
            "extra": "mean: 144.1958815807122 usec\nrounds: 6401"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 8368.318936743795,
            "unit": "iter/sec",
            "range": "stddev: 0.000010974823843348642",
            "extra": "mean: 119.49831352736551 usec\nrounds: 7422"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 7113.146964509247,
            "unit": "iter/sec",
            "range": "stddev: 0.000003192875829538083",
            "extra": "mean: 140.58475172655065 usec\nrounds: 6372"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 8309.519678233406,
            "unit": "iter/sec",
            "range": "stddev: 0.00001013652186699885",
            "extra": "mean: 120.34389937356751 usec\nrounds: 7344"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 6646.802848002588,
            "unit": "iter/sec",
            "range": "stddev: 0.00002176014809042975",
            "extra": "mean: 150.44827157774165 usec\nrounds: 4113"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 812566.9803885582,
            "unit": "iter/sec",
            "range": "stddev: 1.1723303074227942e-7",
            "extra": "mean: 1.2306677777157693 usec\nrounds: 179857"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 19487.298550322113,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021199077672052976",
            "extra": "mean: 51.315475945406014 usec\nrounds: 15361"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 19406.20161335113,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025016690677899677",
            "extra": "mean: 51.52991914254964 usec\nrounds: 17636"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 21586.294507996732,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020432304245979905",
            "extra": "mean: 46.32569057322672 usec\nrounds: 18182"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 19187.45900295757,
            "unit": "iter/sec",
            "range": "stddev: 0.000002640919719621521",
            "extra": "mean: 52.11737520042957 usec\nrounds: 15589"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 21599.584123010623,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015844583625059444",
            "extra": "mean: 46.29718768217731 usec\nrounds: 17844"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 19537.779776287527,
            "unit": "iter/sec",
            "range": "stddev: 0.000002350835995559566",
            "extra": "mean: 51.182888304108786 usec\nrounds: 17673"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 716129.0429518282,
            "unit": "iter/sec",
            "range": "stddev: 2.3014668735271554e-7",
            "extra": "mean: 1.396396375544382 usec\nrounds: 138447"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 276415.7029830662,
            "unit": "iter/sec",
            "range": "stddev: 6.358266322717353e-7",
            "extra": "mean: 3.6177394743064295 usec\nrounds: 108496"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 279983.81685601047,
            "unit": "iter/sec",
            "range": "stddev: 2.778594310888888e-7",
            "extra": "mean: 3.571635001012498 usec\nrounds: 117151"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 281527.66409101465,
            "unit": "iter/sec",
            "range": "stddev: 3.008730997516674e-7",
            "extra": "mean: 3.552048795022544 usec\nrounds: 78840"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 279716.572979083,
            "unit": "iter/sec",
            "range": "stddev: 2.993106674198714e-7",
            "extra": "mean: 3.5750473750970033 usec\nrounds: 51504"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 231095.87278326138,
            "unit": "iter/sec",
            "range": "stddev: 3.6061198492461835e-7",
            "extra": "mean: 4.327208391721791 usec\nrounds: 101338"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 228837.15245259198,
            "unit": "iter/sec",
            "range": "stddev: 3.4007698955079963e-7",
            "extra": "mean: 4.369919784800545 usec\nrounds: 102998"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 202926.2279958455,
            "unit": "iter/sec",
            "range": "stddev: 3.677192351337625e-7",
            "extra": "mean: 4.927899216756116 usec\nrounds: 47875"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 203530.56107405564,
            "unit": "iter/sec",
            "range": "stddev: 3.597214665980422e-7",
            "extra": "mean: 4.913267052981517 usec\nrounds: 62306"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 93505.3774980194,
            "unit": "iter/sec",
            "range": "stddev: 5.137824262425978e-7",
            "extra": "mean: 10.694572085131487 usec\nrounds: 69411"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 98638.87077916678,
            "unit": "iter/sec",
            "range": "stddev: 5.037308927885098e-7",
            "extra": "mean: 10.137991160085411 usec\nrounds: 74661"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 93101.53618816071,
            "unit": "iter/sec",
            "range": "stddev: 0.000001410353844084403",
            "extra": "mean: 10.740961330422873 usec\nrounds: 53634"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 101275.93873228278,
            "unit": "iter/sec",
            "range": "stddev: 5.456322660973336e-7",
            "extra": "mean: 9.874013635592592 usec\nrounds: 61163"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 88642.60885019234,
            "unit": "iter/sec",
            "range": "stddev: 4.98236599927681e-7",
            "extra": "mean: 11.281256418005688 usec\nrounds: 52313"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 95722.54633397907,
            "unit": "iter/sec",
            "range": "stddev: 6.391984049210319e-7",
            "extra": "mean: 10.446859578003362 usec\nrounds: 67169"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 88997.51994642764,
            "unit": "iter/sec",
            "range": "stddev: 6.234437853141119e-7",
            "extra": "mean: 11.236268163449425 usec\nrounds: 52729"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 96573.65727393601,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011370524246207607",
            "extra": "mean: 10.354790615037492 usec\nrounds: 57134"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2821.795443029348,
            "unit": "iter/sec",
            "range": "stddev: 0.00015418867189204182",
            "extra": "mean: 354.3842990002304 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3174.201219951366,
            "unit": "iter/sec",
            "range": "stddev: 0.00001160527467285612",
            "extra": "mean: 315.0398890009001 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2482.8436805142087,
            "unit": "iter/sec",
            "range": "stddev: 0.0015281659984141816",
            "extra": "mean: 402.76397900044003 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3111.755687095161,
            "unit": "iter/sec",
            "range": "stddev: 0.0000074623250233084855",
            "extra": "mean: 321.3619900004119 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 10311.340785209963,
            "unit": "iter/sec",
            "range": "stddev: 0.000004948901597919776",
            "extra": "mean: 96.98059843336249 usec\nrounds: 7660"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 2407.914582616692,
            "unit": "iter/sec",
            "range": "stddev: 0.000017153787712378388",
            "extra": "mean: 415.29712358537876 usec\nrounds: 2209"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1382.361026361066,
            "unit": "iter/sec",
            "range": "stddev: 0.00011639301819812357",
            "extra": "mean: 723.4000242558956 usec\nrounds: 907"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 512926.0721689598,
            "unit": "iter/sec",
            "range": "stddev: 2.5395833793155877e-7",
            "extra": "mean: 1.9495986931828182 usec\nrounds: 119532"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 87597.22223589032,
            "unit": "iter/sec",
            "range": "stddev: 7.00034387546045e-7",
            "extra": "mean: 11.415887107779545 usec\nrounds: 20400"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 70449.843262186,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013291105201767395",
            "extra": "mean: 14.19449573902389 usec\nrounds: 23704"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 56909.73877657621,
            "unit": "iter/sec",
            "range": "stddev: 9.299605130738873e-7",
            "extra": "mean: 17.57168494351964 usec\nrounds: 19314"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 73290.3387396084,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012900180118055096",
            "extra": "mean: 13.644363188890114 usec\nrounds: 14827"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 85229.00028015989,
            "unit": "iter/sec",
            "range": "stddev: 8.207959980862884e-7",
            "extra": "mean: 11.733095504028643 usec\nrounds: 11832"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 42550.982347437566,
            "unit": "iter/sec",
            "range": "stddev: 0.000001833493564172199",
            "extra": "mean: 23.501220061966922 usec\nrounds: 11315"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 16174.268031919251,
            "unit": "iter/sec",
            "range": "stddev: 0.000016314962669103578",
            "extra": "mean: 61.82660000604301 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 51395.258198158735,
            "unit": "iter/sec",
            "range": "stddev: 0.000001770096780112201",
            "extra": "mean: 19.457047888433912 usec\nrounds: 16079"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 24258.000609248167,
            "unit": "iter/sec",
            "range": "stddev: 0.00000382418088892037",
            "extra": "mean: 41.223512856981216 usec\nrounds: 7661"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 27498.43946314862,
            "unit": "iter/sec",
            "range": "stddev: 0.000002283290752325646",
            "extra": "mean: 36.36570000054462 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 15916.061746488485,
            "unit": "iter/sec",
            "range": "stddev: 0.000003342778365213679",
            "extra": "mean: 62.829612998996254 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 11585.838259201406,
            "unit": "iter/sec",
            "range": "stddev: 0.000005505173159200885",
            "extra": "mean: 86.31226999960973 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 19310.769291468863,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036901797919546484",
            "extra": "mean: 51.78457600038655 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 47466.08218639104,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023672396203084925",
            "extra": "mean: 21.067675146922262 usec\nrounds: 19221"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 49651.564331948524,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017820747756304008",
            "extra": "mean: 20.14035234246478 usec\nrounds: 18763"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 7052.322993751913,
            "unit": "iter/sec",
            "range": "stddev: 0.000086077724936058",
            "extra": "mean: 141.79724906048145 usec\nrounds: 3991"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 44150.66454639131,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019975921460821235",
            "extra": "mean: 22.649715701317476 usec\nrounds: 24221"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 32505.40624375273,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028989342498960024",
            "extra": "mean: 30.764113283223207 usec\nrounds: 17805"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 38977.958992547916,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026229918779850207",
            "extra": "mean: 25.655524964536678 usec\nrounds: 11957"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 45696.72834349878,
            "unit": "iter/sec",
            "range": "stddev: 0.000001998474981270643",
            "extra": "mean: 21.88340470422909 usec\nrounds: 17005"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 38352.11205458569,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020533408637197595",
            "extra": "mean: 26.074183309037135 usec\nrounds: 15062"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 24421.796989144692,
            "unit": "iter/sec",
            "range": "stddev: 0.000002635024241928371",
            "extra": "mean: 40.94702779015372 usec\nrounds: 11119"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 16092.25529438581,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026703554396242456",
            "extra": "mean: 62.14169373443108 usec\nrounds: 11108"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 8012.664811970956,
            "unit": "iter/sec",
            "range": "stddev: 0.0000054561639623927775",
            "extra": "mean: 124.802425094083 usec\nrounds: 5587"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2456.737073395718,
            "unit": "iter/sec",
            "range": "stddev: 0.00000859897666856399",
            "extra": "mean: 407.04396527781194 usec\nrounds: 2160"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 906564.8281636839,
            "unit": "iter/sec",
            "range": "stddev: 8.40142819142614e-8",
            "extra": "mean: 1.103065074811734 usec\nrounds: 170911"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3672.817797610647,
            "unit": "iter/sec",
            "range": "stddev: 0.00006599059375938602",
            "extra": "mean: 272.27051683602446 usec\nrounds: 2465"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 3867.7798312361183,
            "unit": "iter/sec",
            "range": "stddev: 0.0000073682336726119485",
            "extra": "mean: 258.54625744827007 usec\nrounds: 3457"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 940.8473043666198,
            "unit": "iter/sec",
            "range": "stddev: 0.005274328345074609",
            "extra": "mean: 1.0628717278126252 msec\nrounds: 169"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 608.9718014372628,
            "unit": "iter/sec",
            "range": "stddev: 0.00281757963710729",
            "extra": "mean: 1.6421121596104338 msec\nrounds: 614"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 610.3118009106014,
            "unit": "iter/sec",
            "range": "stddev: 0.0026270973772948144",
            "extra": "mean: 1.638506741157509 msec\nrounds: 622"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 9580.569199790794,
            "unit": "iter/sec",
            "range": "stddev: 0.0000070622350664541925",
            "extra": "mean: 104.37793195229324 usec\nrounds: 2557"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 11782.899808398577,
            "unit": "iter/sec",
            "range": "stddev: 0.00004900850477907273",
            "extra": "mean: 84.86875185743524 usec\nrounds: 8479"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 187902.9358255091,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017560814221820561",
            "extra": "mean: 5.321896625014004 usec\nrounds: 23052"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1567801.430213052,
            "unit": "iter/sec",
            "range": "stddev: 5.765437544900155e-8",
            "extra": "mean: 637.835876871287 nsec\nrounds: 71757"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 285180.5678644006,
            "unit": "iter/sec",
            "range": "stddev: 5.594710988572299e-7",
            "extra": "mean: 3.5065502796652193 usec\nrounds: 26681"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 4000.6107705771096,
            "unit": "iter/sec",
            "range": "stddev: 0.00005574604419986496",
            "extra": "mean: 249.96183266680166 usec\nrounds: 1751"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 3107.812995702234,
            "unit": "iter/sec",
            "range": "stddev: 0.0000530879697495663",
            "extra": "mean: 321.7696822115394 usec\nrounds: 2659"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1605.369790295545,
            "unit": "iter/sec",
            "range": "stddev: 0.000060413003981547594",
            "extra": "mean: 622.909441827669 usec\nrounds: 1444"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 8.6366626735186,
            "unit": "iter/sec",
            "range": "stddev: 0.02326492099609599",
            "extra": "mean: 115.78546457142076 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 2.6164720385734475,
            "unit": "iter/sec",
            "range": "stddev: 0.08729451869750092",
            "extra": "mean: 382.1940327500002 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.030050705032414,
            "unit": "iter/sec",
            "range": "stddev: 0.022412952183219717",
            "extra": "mean: 110.74134937500446 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 8.114744386797648,
            "unit": "iter/sec",
            "range": "stddev: 0.007033301110539479",
            "extra": "mean: 123.23247071429121 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 2.4494845675772483,
            "unit": "iter/sec",
            "range": "stddev: 0.08861663142747114",
            "extra": "mean: 408.24915300000697 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_filesystem",
            "value": 9.619791451097337,
            "unit": "iter/sec",
            "range": "stddev: 0.020733405360617845",
            "extra": "mean: 103.95235750000893 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_filesystem",
            "value": 10.013809771951012,
            "unit": "iter/sec",
            "range": "stddev: 0.021365257570953422",
            "extra": "mean: 99.86209272728854 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 38934.52330391446,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010983964097949605",
            "extra": "mean: 25.684146488560206 usec\nrounds: 9284"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 30166.71726913043,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017360390108629788",
            "extra": "mean: 33.14911566540582 usec\nrounds: 8075"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 39275.9086372112,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014839599806500777",
            "extra": "mean: 25.460900452665005 usec\nrounds: 14576"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 26901.68229771848,
            "unit": "iter/sec",
            "range": "stddev: 0.000001932676152057456",
            "extra": "mean: 37.17239646699752 usec\nrounds: 10417"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 33383.03162119816,
            "unit": "iter/sec",
            "range": "stddev: 0.000002927016057472546",
            "extra": "mean: 29.955338129476587 usec\nrounds: 10981"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 25691.304310331478,
            "unit": "iter/sec",
            "range": "stddev: 0.000003953044888201986",
            "extra": "mean: 38.923675805663976 usec\nrounds: 11672"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 33539.115868228306,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019365481155761604",
            "extra": "mean: 29.81593205762775 usec\nrounds: 8360"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 23939.500208048954,
            "unit": "iter/sec",
            "range": "stddev: 0.000002424666492861481",
            "extra": "mean: 41.77196647003429 usec\nrounds: 8798"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 28499.724746072137,
            "unit": "iter/sec",
            "range": "stddev: 0.000002610257468287681",
            "extra": "mean: 35.08805817985387 usec\nrounds: 8594"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 17634.45735446301,
            "unit": "iter/sec",
            "range": "stddev: 0.00000490089470825734",
            "extra": "mean: 56.70716029982716 usec\nrounds: 8141"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 9790.311309907105,
            "unit": "iter/sec",
            "range": "stddev: 0.000008677791829390285",
            "extra": "mean: 102.1417979822634 usec\nrounds: 3965"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 53276.57318330028,
            "unit": "iter/sec",
            "range": "stddev: 0.000002072488319197609",
            "extra": "mean: 18.769975999759936 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 27650.895499284194,
            "unit": "iter/sec",
            "range": "stddev: 0.000002179965091223239",
            "extra": "mean: 36.16519399980689 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 46466.44091432622,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013331711638547699",
            "extra": "mean: 21.52090799989992 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 23764.093652180985,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024927525721848742",
            "extra": "mean: 42.080292000036934 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 765.0469970664685,
            "unit": "iter/sec",
            "range": "stddev: 0.000024490595271356012",
            "extra": "mean: 1.3071092414380374 msec\nrounds: 584"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 990.6420431520565,
            "unit": "iter/sec",
            "range": "stddev: 0.00008882921821205338",
            "extra": "mean: 1.0094463554344697 msec\nrounds: 920"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6295.419663396195,
            "unit": "iter/sec",
            "range": "stddev: 0.0000068865504376992985",
            "extra": "mean: 158.84564548005517 usec\nrounds: 2533"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 6664.349759567985,
            "unit": "iter/sec",
            "range": "stddev: 0.000010755505406474331",
            "extra": "mean: 150.05214853321635 usec\nrounds: 3851"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 6770.079191521463,
            "unit": "iter/sec",
            "range": "stddev: 0.000007381328235499457",
            "extra": "mean: 147.7087596334699 usec\nrounds: 2180"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4485.6771935607,
            "unit": "iter/sec",
            "range": "stddev: 0.000013130467905738159",
            "extra": "mean: 222.93177971779258 usec\nrounds: 2692"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 2084.772161478036,
            "unit": "iter/sec",
            "range": "stddev: 0.00001943492806155089",
            "extra": "mean: 479.66872278792914 usec\nrounds: 1865"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 2012.77765602322,
            "unit": "iter/sec",
            "range": "stddev: 0.00006618191597984442",
            "extra": "mean: 496.82586499681594 usec\nrounds: 1637"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 2592.5434704988625,
            "unit": "iter/sec",
            "range": "stddev: 0.00001700732331828328",
            "extra": "mean: 385.7215940173138 usec\nrounds: 2340"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 2557.436487300698,
            "unit": "iter/sec",
            "range": "stddev: 0.000034972063175448024",
            "extra": "mean: 391.01655308573146 usec\nrounds: 2430"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1824.4346767825793,
            "unit": "iter/sec",
            "range": "stddev: 0.00002324072919310818",
            "extra": "mean: 548.1149929486741 usec\nrounds: 1702"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1818.2239923239365,
            "unit": "iter/sec",
            "range": "stddev: 0.000033939313124549335",
            "extra": "mean: 549.9872426179265 usec\nrounds: 1727"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 2231.839848214079,
            "unit": "iter/sec",
            "range": "stddev: 0.00001448707265557806",
            "extra": "mean: 448.060823360691 usec\nrounds: 2089"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 2166.787381512441,
            "unit": "iter/sec",
            "range": "stddev: 0.00006610367597499538",
            "extra": "mean: 461.5127485660311 usec\nrounds: 2092"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 582.5567607546513,
            "unit": "iter/sec",
            "range": "stddev: 0.000021488561438009014",
            "extra": "mean: 1.716570928993404 msec\nrounds: 507"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 591.5228374631942,
            "unit": "iter/sec",
            "range": "stddev: 0.000017513369987361935",
            "extra": "mean: 1.690551804032794 msec\nrounds: 347"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6f2907205f7638575690e2a376cdee33821c28b1",
          "message": "[BugFix] Fix functorch dim mock (#777)",
          "timestamp": "2024-05-14T17:03:46+01:00",
          "tree_id": "57f2db169a1f1ffaae324b33349a757a62fb1626",
          "url": "https://github.com/pytorch/tensordict/commit/6f2907205f7638575690e2a376cdee33821c28b1"
        },
        "date": 1715702891707,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 59388.983225490876,
            "unit": "iter/sec",
            "range": "stddev: 7.394708235668146e-7",
            "extra": "mean: 16.838139764123476 usec\nrounds: 8221"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 57963.62619730598,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010239703691391546",
            "extra": "mean: 17.252198759891904 usec\nrounds: 17735"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 52269.53594818122,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017587157305775684",
            "extra": "mean: 19.13160279424283 usec\nrounds: 31636"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 52442.70448595523,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013262984492477894",
            "extra": "mean: 19.068429246775622 usec\nrounds: 35320"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 398284.011453564,
            "unit": "iter/sec",
            "range": "stddev: 1.9622561244548207e-7",
            "extra": "mean: 2.510771136281453 usec\nrounds: 42510"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 3756.98466704093,
            "unit": "iter/sec",
            "range": "stddev: 0.000019707518735213075",
            "extra": "mean: 266.1709026317689 usec\nrounds: 3153"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 3776.7426359767833,
            "unit": "iter/sec",
            "range": "stddev: 0.000011714698553592644",
            "extra": "mean: 264.77843379480606 usec\nrounds: 3610"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 12980.302591735524,
            "unit": "iter/sec",
            "range": "stddev: 0.000004331149300601384",
            "extra": "mean: 77.03980650163685 usec\nrounds: 9752"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 3765.6289728026245,
            "unit": "iter/sec",
            "range": "stddev: 0.000007463600152768291",
            "extra": "mean: 265.5598858046111 usec\nrounds: 3494"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 13166.617191710286,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018302119249057619",
            "extra": "mean: 75.94965247638558 usec\nrounds: 11205"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 3745.109599480648,
            "unit": "iter/sec",
            "range": "stddev: 0.000005129971831247931",
            "extra": "mean: 267.0148825921342 usec\nrounds: 3688"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 260688.55898104224,
            "unit": "iter/sec",
            "range": "stddev: 2.700218867952176e-7",
            "extra": "mean: 3.835994966210704 usec\nrounds: 118400"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 7060.2310551586925,
            "unit": "iter/sec",
            "range": "stddev: 0.000003297310719487068",
            "extra": "mean: 141.63842403844998 usec\nrounds: 6082"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 6895.229611928996,
            "unit": "iter/sec",
            "range": "stddev: 0.000021662105276876267",
            "extra": "mean: 145.0278027391523 usec\nrounds: 6352"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 8420.399137191973,
            "unit": "iter/sec",
            "range": "stddev: 0.000004969816043691777",
            "extra": "mean: 118.75921600712614 usec\nrounds: 7509"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 7240.364745993196,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036905874978630115",
            "extra": "mean: 138.11458884766793 usec\nrounds: 6438"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 8498.91210149258,
            "unit": "iter/sec",
            "range": "stddev: 0.000005361720358089437",
            "extra": "mean: 117.66211816973372 usec\nrounds: 7540"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 7049.173337152021,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034603962923060153",
            "extra": "mean: 141.8606058003414 usec\nrounds: 6517"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 865891.5143705944,
            "unit": "iter/sec",
            "range": "stddev: 7.056974307911822e-8",
            "extra": "mean: 1.154879085201438 usec\nrounds: 189036"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 19995.562807538932,
            "unit": "iter/sec",
            "range": "stddev: 0.000002648198317117688",
            "extra": "mean: 50.01109544278342 usec\nrounds: 14878"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 19835.635186075448,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015929201939951406",
            "extra": "mean: 50.41431699157266 usec\nrounds: 17991"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 21905.688417781417,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025024566240470213",
            "extra": "mean: 45.65024302948973 usec\nrounds: 18076"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 19284.0947501214,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021387103065973355",
            "extra": "mean: 51.856206524483326 usec\nrounds: 15480"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 22277.668162611277,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015231451363503944",
            "extra": "mean: 44.888001414721906 usec\nrounds: 17672"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 19349.781195020554,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030221096886190606",
            "extra": "mean: 51.68017094980581 usec\nrounds: 17432"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 754586.3327477477,
            "unit": "iter/sec",
            "range": "stddev: 1.834320453547712e-7",
            "extra": "mean: 1.3252294092825723 usec\nrounds: 163908"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 291583.9890590816,
            "unit": "iter/sec",
            "range": "stddev: 3.0539765778619655e-7",
            "extra": "mean: 3.429543587859267 usec\nrounds: 117028"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 281037.8638226963,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014950558282607205",
            "extra": "mean: 3.5582394001930253 usec\nrounds: 118400"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 293623.8210364968,
            "unit": "iter/sec",
            "range": "stddev: 7.375546869764873e-7",
            "extra": "mean: 3.405718229774356 usec\nrounds: 52699"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 291166.8245969885,
            "unit": "iter/sec",
            "range": "stddev: 5.565044768997256e-7",
            "extra": "mean: 3.434457209828509 usec\nrounds: 54452"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 236226.75518193372,
            "unit": "iter/sec",
            "range": "stddev: 5.854546655289812e-7",
            "extra": "mean: 4.233220742628557 usec\nrounds: 100624"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 241220.0190201122,
            "unit": "iter/sec",
            "range": "stddev: 3.8394932339986713e-7",
            "extra": "mean: 4.145592907513298 usec\nrounds: 108850"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 74992.83062108589,
            "unit": "iter/sec",
            "range": "stddev: 8.825573965580184e-7",
            "extra": "mean: 13.334608011433401 usec\nrounds: 37996"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 74940.14532798137,
            "unit": "iter/sec",
            "range": "stddev: 7.622231938635535e-7",
            "extra": "mean: 13.343982662742677 usec\nrounds: 38126"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 94121.62099700548,
            "unit": "iter/sec",
            "range": "stddev: 4.895677224915099e-7",
            "extra": "mean: 10.624551398576267 usec\nrounds: 66850"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 98929.58100334024,
            "unit": "iter/sec",
            "range": "stddev: 5.145029189297861e-7",
            "extra": "mean: 10.108200094026843 usec\nrounds: 74655"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 93536.50981795654,
            "unit": "iter/sec",
            "range": "stddev: 6.649372688309839e-7",
            "extra": "mean: 10.691012546290523 usec\nrounds: 39296"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 98915.89121230373,
            "unit": "iter/sec",
            "range": "stddev: 4.872141812536377e-7",
            "extra": "mean: 10.109599051720561 usec\nrounds: 59060"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 89500.07519265081,
            "unit": "iter/sec",
            "range": "stddev: 5.723379614883869e-7",
            "extra": "mean: 11.173174970495598 usec\nrounds: 52203"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 95942.68434540307,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010479096030087854",
            "extra": "mean: 10.422889528501223 usec\nrounds: 68597"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 89517.87027306065,
            "unit": "iter/sec",
            "range": "stddev: 6.460429121816569e-7",
            "extra": "mean: 11.170953877138185 usec\nrounds: 53119"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 96702.23196465781,
            "unit": "iter/sec",
            "range": "stddev: 5.219918132410536e-7",
            "extra": "mean: 10.3410229493511 usec\nrounds: 58956"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2857.2227287816227,
            "unit": "iter/sec",
            "range": "stddev: 0.00013304551197813899",
            "extra": "mean: 349.9902159977637 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3321.0546084829034,
            "unit": "iter/sec",
            "range": "stddev: 0.00001614485586615039",
            "extra": "mean: 301.10917099818835 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2485.7705791926783,
            "unit": "iter/sec",
            "range": "stddev: 0.001658981762788627",
            "extra": "mean: 402.28973999876416 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3258.5602165280025,
            "unit": "iter/sec",
            "range": "stddev: 0.00000894775800860594",
            "extra": "mean: 306.88400199812804 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 10225.613620251654,
            "unit": "iter/sec",
            "range": "stddev: 0.000007172810871007954",
            "extra": "mean: 97.79364223380365 usec\nrounds: 7558"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 2442.1830649706167,
            "unit": "iter/sec",
            "range": "stddev: 0.000007806888113583807",
            "extra": "mean: 409.46971352945303 usec\nrounds: 2269"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1435.5502427149347,
            "unit": "iter/sec",
            "range": "stddev: 0.00006825960495850789",
            "extra": "mean: 696.5970052770739 usec\nrounds: 947"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 515727.751629316,
            "unit": "iter/sec",
            "range": "stddev: 2.424543092969918e-7",
            "extra": "mean: 1.9390075419458115 usec\nrounds: 118540"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 99858.10089002704,
            "unit": "iter/sec",
            "range": "stddev: 8.823276490065249e-7",
            "extra": "mean: 10.014210074967203 usec\nrounds: 933"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 79239.86773349403,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010388088515667778",
            "extra": "mean: 12.619910009987414 usec\nrounds: 27803"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 61641.895426829265,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013835437878341907",
            "extra": "mean: 16.222732819548504 usec\nrounds: 24897"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 74099.11460492454,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015456006624423864",
            "extra": "mean: 13.495437905455635 usec\nrounds: 14816"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 86258.72177749983,
            "unit": "iter/sec",
            "range": "stddev: 8.123333708849543e-7",
            "extra": "mean: 11.593030587439625 usec\nrounds: 10854"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 43168.48625312643,
            "unit": "iter/sec",
            "range": "stddev: 0.000001840914629377289",
            "extra": "mean: 23.16504669949079 usec\nrounds: 10814"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 17085.197043857548,
            "unit": "iter/sec",
            "range": "stddev: 0.000012117350138352654",
            "extra": "mean: 58.53019999904063 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 51225.62019887417,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020473701364164147",
            "extra": "mean: 19.52148155781583 usec\nrounds: 15996"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 24167.067369388842,
            "unit": "iter/sec",
            "range": "stddev: 0.000004458501623826293",
            "extra": "mean: 41.378624254039515 usec\nrounds: 7537"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 29026.418513392116,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023911758289512",
            "extra": "mean: 34.451373997058 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 16475.63714643128,
            "unit": "iter/sec",
            "range": "stddev: 0.000003085288582916117",
            "extra": "mean: 60.69567999782066 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 11711.590375426587,
            "unit": "iter/sec",
            "range": "stddev: 0.000005016176719325314",
            "extra": "mean: 85.38549999991574 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 19729.37492554243,
            "unit": "iter/sec",
            "range": "stddev: 0.000003683977541525972",
            "extra": "mean: 50.685843001815556 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 49924.11572903165,
            "unit": "iter/sec",
            "range": "stddev: 0.000001727898274608605",
            "extra": "mean: 20.030399845790047 usec\nrounds: 16889"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 50472.98979885467,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023002722338946216",
            "extra": "mean: 19.812577063201672 usec\nrounds: 18355"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 6946.917397842651,
            "unit": "iter/sec",
            "range": "stddev: 0.0000869954965083859",
            "extra": "mean: 143.94873909261506 usec\nrounds: 3599"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 46708.05480238753,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019180820243838477",
            "extra": "mean: 21.40958351253977 usec\nrounds: 25547"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 33684.624540498495,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026959864162392207",
            "extra": "mean: 29.687135114055245 usec\nrounds: 17748"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 39361.853448753354,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018265500966613363",
            "extra": "mean: 25.405307737907638 usec\nrounds: 11747"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 45872.23235987782,
            "unit": "iter/sec",
            "range": "stddev: 0.00000258491371701344",
            "extra": "mean: 21.79968029798024 usec\nrounds: 16531"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 38656.239303933195,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022008222588905783",
            "extra": "mean: 25.869045153035675 usec\nrounds: 14927"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 24784.077227205005,
            "unit": "iter/sec",
            "range": "stddev: 0.000002718618766600397",
            "extra": "mean: 40.34848628143876 usec\nrounds: 10788"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 16628.50083053637,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020328951533208664",
            "extra": "mean: 60.13771236452132 usec\nrounds: 11452"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 8084.340864294405,
            "unit": "iter/sec",
            "range": "stddev: 0.000007645861808582241",
            "extra": "mean: 123.69592237465352 usec\nrounds: 5694"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2515.8650554432043,
            "unit": "iter/sec",
            "range": "stddev: 0.000007877381696589747",
            "extra": "mean: 397.4775983459241 usec\nrounds: 2176"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 930617.5763540439,
            "unit": "iter/sec",
            "range": "stddev: 8.217706116542954e-8",
            "extra": "mean: 1.0745552474065283 usec\nrounds: 178859"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3833.170369664207,
            "unit": "iter/sec",
            "range": "stddev: 0.000045973550551421355",
            "extra": "mean: 260.88065584405575 usec\nrounds: 2371"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 4047.401890089397,
            "unit": "iter/sec",
            "range": "stddev: 0.000007437094390548485",
            "extra": "mean: 247.07207911540323 usec\nrounds: 3299"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1331.1565077336843,
            "unit": "iter/sec",
            "range": "stddev: 0.003144515002640126",
            "extra": "mean: 751.2264667529713 usec\nrounds: 1534"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 581.708037045498,
            "unit": "iter/sec",
            "range": "stddev: 0.004615358635220363",
            "extra": "mean: 1.719075440454651 msec\nrounds: 613"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 617.4511020372848,
            "unit": "iter/sec",
            "range": "stddev: 0.002653018174171102",
            "extra": "mean: 1.6195614465671728 msec\nrounds: 627"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 9677.15821704086,
            "unit": "iter/sec",
            "range": "stddev: 0.000008788190274052495",
            "extra": "mean: 103.33612177995225 usec\nrounds: 2833"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 12043.760030816284,
            "unit": "iter/sec",
            "range": "stddev: 0.000035414989532063335",
            "extra": "mean: 83.03054838699101 usec\nrounds: 8494"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 185493.28160746253,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016648468065422384",
            "extra": "mean: 5.391030830519143 usec\nrounds: 23256"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1581642.8634181735,
            "unit": "iter/sec",
            "range": "stddev: 1.7514719668400507e-7",
            "extra": "mean: 632.2539829496314 nsec\nrounds: 76139"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 280248.53153225774,
            "unit": "iter/sec",
            "range": "stddev: 3.63247050542962e-7",
            "extra": "mean: 3.56826133765092 usec\nrounds: 27497"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 3992.7290753990187,
            "unit": "iter/sec",
            "range": "stddev: 0.00005414301812537249",
            "extra": "mean: 250.45526032844182 usec\nrounds: 1694"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 3081.735757811053,
            "unit": "iter/sec",
            "range": "stddev: 0.00005105632443218172",
            "extra": "mean: 324.4924544440166 usec\nrounds: 2689"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1679.7651213553231,
            "unit": "iter/sec",
            "range": "stddev: 0.00005962373906396628",
            "extra": "mean: 595.3213263489762 usec\nrounds: 1575"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 8.92794643246549,
            "unit": "iter/sec",
            "range": "stddev: 0.024001978061729164",
            "extra": "mean: 112.00784049998447 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 2.5989893538160334,
            "unit": "iter/sec",
            "range": "stddev: 0.0890501413472702",
            "extra": "mean: 384.7649466250118 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 8.507194838808404,
            "unit": "iter/sec",
            "range": "stddev: 0.022377442514044384",
            "extra": "mean: 117.54756049998605 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 7.295563698409877,
            "unit": "iter/sec",
            "range": "stddev: 0.03194680164507784",
            "extra": "mean: 137.06960028571302 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 2.4382967166960166,
            "unit": "iter/sec",
            "range": "stddev: 0.07267834857165996",
            "extra": "mean: 410.12235842856626 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_filesystem",
            "value": 10.754534934895691,
            "unit": "iter/sec",
            "range": "stddev: 0.0055687436767148145",
            "extra": "mean: 92.98403009090221 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_filesystem",
            "value": 9.745959813575485,
            "unit": "iter/sec",
            "range": "stddev: 0.022493778478647284",
            "extra": "mean: 102.60662050002149 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 38720.812950387175,
            "unit": "iter/sec",
            "range": "stddev: 0.000001189197769571426",
            "extra": "mean: 25.825904050136966 usec\nrounds: 7556"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 29871.149416658976,
            "unit": "iter/sec",
            "range": "stddev: 0.000001747956352697435",
            "extra": "mean: 33.47711820698488 usec\nrounds: 8079"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 39041.519870961856,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016302173191499914",
            "extra": "mean: 25.613756926091803 usec\nrounds: 12741"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 26421.387124927158,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021375700028427054",
            "extra": "mean: 37.84812641636645 usec\nrounds: 8646"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 34142.63803702257,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013917038413022298",
            "extra": "mean: 29.288890885222465 usec\nrounds: 11300"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 24842.925071076603,
            "unit": "iter/sec",
            "range": "stddev: 0.000009096768667198222",
            "extra": "mean: 40.25290891225409 usec\nrounds: 11725"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 34380.01294723449,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017807322001434944",
            "extra": "mean: 29.086667347530458 usec\nrounds: 7840"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 24240.102148799946,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024775359360711507",
            "extra": "mean: 41.25395156593872 usec\nrounds: 10117"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 29560.310559864,
            "unit": "iter/sec",
            "range": "stddev: 0.000002319816010420831",
            "extra": "mean: 33.82914391155844 usec\nrounds: 7866"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 18346.41006705281,
            "unit": "iter/sec",
            "range": "stddev: 0.000005530100201066441",
            "extra": "mean: 54.50657629177485 usec\nrounds: 7989"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 10033.951202986334,
            "unit": "iter/sec",
            "range": "stddev: 0.000007701304194901903",
            "extra": "mean: 99.66163675406125 usec\nrounds: 3956"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 57040.907105523234,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012704500091772548",
            "extra": "mean: 17.531278002820727 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 28730.660457949743,
            "unit": "iter/sec",
            "range": "stddev: 0.000001787243400452785",
            "extra": "mean: 34.80602200090743 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 49098.85435328208,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013274462451053945",
            "extra": "mean: 20.367074001455876 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 25231.530834426318,
            "unit": "iter/sec",
            "range": "stddev: 0.00000197878099635045",
            "extra": "mean: 39.63295000062317 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 761.2412259596379,
            "unit": "iter/sec",
            "range": "stddev: 0.00002531108739705061",
            "extra": "mean: 1.313644040677615 msec\nrounds: 590"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 984.7967009039676,
            "unit": "iter/sec",
            "range": "stddev: 0.00009774618398741777",
            "extra": "mean: 1.01543800774523 msec\nrounds: 904"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6107.1518844240545,
            "unit": "iter/sec",
            "range": "stddev: 0.0000068197287962055156",
            "extra": "mean: 163.74244802236595 usec\nrounds: 2453"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 6678.447118495118,
            "unit": "iter/sec",
            "range": "stddev: 0.0000070116787273045005",
            "extra": "mean: 149.7354073869397 usec\nrounds: 3574"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 6835.92788030229,
            "unit": "iter/sec",
            "range": "stddev: 0.000006834441600794704",
            "extra": "mean: 146.28592014282327 usec\nrounds: 1966"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4482.842359986122,
            "unit": "iter/sec",
            "range": "stddev: 0.000022713017497097257",
            "extra": "mean: 223.07275600989365 usec\nrounds: 2496"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 2101.545635522968,
            "unit": "iter/sec",
            "range": "stddev: 0.000017415649291090195",
            "extra": "mean: 475.8402497175137 usec\nrounds: 1766"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 2105.746870938894,
            "unit": "iter/sec",
            "range": "stddev: 0.000016469756530140272",
            "extra": "mean: 474.890887314546 usec\nrounds: 1553"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 2557.234297424306,
            "unit": "iter/sec",
            "range": "stddev: 0.000015472687545815176",
            "extra": "mean: 391.04746913773937 usec\nrounds: 2430"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 2566.282554317441,
            "unit": "iter/sec",
            "range": "stddev: 0.00001624049374312879",
            "extra": "mean: 389.6687051539311 usec\nrounds: 2425"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1824.8719487694923,
            "unit": "iter/sec",
            "range": "stddev: 0.000021664977147592293",
            "extra": "mean: 547.9836547842702 usec\nrounds: 1599"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1803.071466985071,
            "unit": "iter/sec",
            "range": "stddev: 0.000026564246285974706",
            "extra": "mean: 554.6091867740037 usec\nrounds: 1724"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 2188.376010128527,
            "unit": "iter/sec",
            "range": "stddev: 0.000019391775849910028",
            "extra": "mean: 456.9598621862375 usec\nrounds: 2039"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 2215.7903450377166,
            "unit": "iter/sec",
            "range": "stddev: 0.00001453230228579837",
            "extra": "mean: 451.30623582664725 usec\nrounds: 2099"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 584.0262788644135,
            "unit": "iter/sec",
            "range": "stddev: 0.00008462540617025387",
            "extra": "mean: 1.7122517191253963 msec\nrounds: 502"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 594.3685502414553,
            "unit": "iter/sec",
            "range": "stddev: 0.000022786101752261624",
            "extra": "mean: 1.682457794231814 msec\nrounds: 520"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5fef53825ae0034b442ba6e3bfbaa6933625fa71",
          "message": "[Feature] online edition of memory mapped tensordicts (#775)",
          "timestamp": "2024-05-14T18:03:22+01:00",
          "tree_id": "26e5300a78f6d4a498d8f887f72214047cf6dd02",
          "url": "https://github.com/pytorch/tensordict/commit/5fef53825ae0034b442ba6e3bfbaa6933625fa71"
        },
        "date": 1715706483540,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 61496.33420043609,
            "unit": "iter/sec",
            "range": "stddev: 6.396095080765203e-7",
            "extra": "mean: 16.261131870733664 usec\nrounds: 8296"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 60864.42185886501,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011293164757001645",
            "extra": "mean: 16.42995972784958 usec\nrounds: 17183"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 54216.21686600212,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011631409260265392",
            "extra": "mean: 18.44466578093315 usec\nrounds: 32365"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 53929.094170654775,
            "unit": "iter/sec",
            "range": "stddev: 0.000001432513958512076",
            "extra": "mean: 18.542866617332216 usec\nrounds: 36309"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 393370.32232882484,
            "unit": "iter/sec",
            "range": "stddev: 1.9367379529751483e-7",
            "extra": "mean: 2.5421338195515504 usec\nrounds: 72538"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 3703.8756942798577,
            "unit": "iter/sec",
            "range": "stddev: 0.000025131668273230373",
            "extra": "mean: 269.9874624692094 usec\nrounds: 3224"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 3724.2729840907355,
            "unit": "iter/sec",
            "range": "stddev: 0.00001164391315997543",
            "extra": "mean: 268.5087812498646 usec\nrounds: 3648"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 12695.904931496243,
            "unit": "iter/sec",
            "range": "stddev: 0.000005496727425430781",
            "extra": "mean: 78.76555514520126 usec\nrounds: 10019"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 3720.6211040278463,
            "unit": "iter/sec",
            "range": "stddev: 0.000006116919906614598",
            "extra": "mean: 268.7723291461811 usec\nrounds: 3503"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 12396.88701124971,
            "unit": "iter/sec",
            "range": "stddev: 0.000007247722506568055",
            "extra": "mean: 80.66541213875206 usec\nrounds: 10858"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 3671.4050587416536,
            "unit": "iter/sec",
            "range": "stddev: 0.00003769659934012734",
            "extra": "mean: 272.3752852110365 usec\nrounds: 3692"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 263913.1918197661,
            "unit": "iter/sec",
            "range": "stddev: 2.6394067602855304e-7",
            "extra": "mean: 3.789124723567925 usec\nrounds: 113547"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 7069.7644520292515,
            "unit": "iter/sec",
            "range": "stddev: 0.000006307816208749961",
            "extra": "mean: 141.44742823970148 usec\nrounds: 6020"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 6826.007369906795,
            "unit": "iter/sec",
            "range": "stddev: 0.000026400753321673448",
            "extra": "mean: 146.49852334010217 usec\nrounds: 6384"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 8291.306076846959,
            "unit": "iter/sec",
            "range": "stddev: 0.0000057042750998833395",
            "extra": "mean: 120.60826011386169 usec\nrounds: 7366"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 6954.28005053731,
            "unit": "iter/sec",
            "range": "stddev: 0.000003335210300439005",
            "extra": "mean: 143.79633732506025 usec\nrounds: 6347"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 8256.748153462939,
            "unit": "iter/sec",
            "range": "stddev: 0.000003279103505619926",
            "extra": "mean: 121.11305581975307 usec\nrounds: 7345"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 6876.626439704527,
            "unit": "iter/sec",
            "range": "stddev: 0.000003482782920983804",
            "extra": "mean: 145.4201429680929 usec\nrounds: 6449"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 802178.207431912,
            "unit": "iter/sec",
            "range": "stddev: 7.083592230892595e-8",
            "extra": "mean: 1.2466057924976464 usec\nrounds: 194553"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 19925.377062096846,
            "unit": "iter/sec",
            "range": "stddev: 0.000001882036120638816",
            "extra": "mean: 50.187256024492264 usec\nrounds: 15022"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 19705.113389487633,
            "unit": "iter/sec",
            "range": "stddev: 0.000029635831130302778",
            "extra": "mean: 50.74824895620668 usec\nrounds: 17959"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 21915.74796680737,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018010965434586563",
            "extra": "mean: 45.629289108204574 usec\nrounds: 18142"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 19387.646087199566,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016351435376806518",
            "extra": "mean: 51.57923739180677 usec\nrounds: 15565"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 21772.706413391355,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019859563557496846",
            "extra": "mean: 45.929062791428976 usec\nrounds: 17789"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 19256.12833391563,
            "unit": "iter/sec",
            "range": "stddev: 0.00000182293388548864",
            "extra": "mean: 51.93151928878194 usec\nrounds: 17549"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 739187.2184807122,
            "unit": "iter/sec",
            "range": "stddev: 1.929191625635517e-7",
            "extra": "mean: 1.352837244744774 usec\nrounds: 169463"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 291440.9151884867,
            "unit": "iter/sec",
            "range": "stddev: 3.1558874682502744e-7",
            "extra": "mean: 3.431227215826094 usec\nrounds: 108496"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 288423.3062736236,
            "unit": "iter/sec",
            "range": "stddev: 3.810124173457754e-7",
            "extra": "mean: 3.4671261935098703 usec\nrounds: 122011"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 289596.4187292866,
            "unit": "iter/sec",
            "range": "stddev: 3.9927904359657815e-7",
            "extra": "mean: 3.453081375756913 usec\nrounds: 84374"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 290386.91744914633,
            "unit": "iter/sec",
            "range": "stddev: 3.1312358655432433e-7",
            "extra": "mean: 3.4436813090077445 usec\nrounds: 48103"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 238674.44383948162,
            "unit": "iter/sec",
            "range": "stddev: 3.6543103448541224e-7",
            "extra": "mean: 4.189807605344378 usec\nrounds: 100221"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 227140.73391729465,
            "unit": "iter/sec",
            "range": "stddev: 9.35021676296822e-7",
            "extra": "mean: 4.402556876320189 usec\nrounds: 108850"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 239306.0137472272,
            "unit": "iter/sec",
            "range": "stddev: 4.283727104063388e-7",
            "extra": "mean: 4.17874997933096 usec\nrounds: 60383"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 237701.48171673625,
            "unit": "iter/sec",
            "range": "stddev: 3.2781544925902055e-7",
            "extra": "mean: 4.206957368451235 usec\nrounds: 75273"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 95160.38241865856,
            "unit": "iter/sec",
            "range": "stddev: 4.976045414998759e-7",
            "extra": "mean: 10.508574835276463 usec\nrounds: 70742"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 99335.25550871511,
            "unit": "iter/sec",
            "range": "stddev: 8.114088196111309e-7",
            "extra": "mean: 10.066919291431889 usec\nrounds: 76547"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 94567.05745271248,
            "unit": "iter/sec",
            "range": "stddev: 7.052675910458148e-7",
            "extra": "mean: 10.574506883647533 usec\nrounds: 58473"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 99761.18377861832,
            "unit": "iter/sec",
            "range": "stddev: 8.724390423577774e-7",
            "extra": "mean: 10.023938791856324 usec\nrounds: 59698"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 89726.73971858874,
            "unit": "iter/sec",
            "range": "stddev: 6.612284352325832e-7",
            "extra": "mean: 11.14494968987299 usec\nrounds: 46949"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 95320.38591291317,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012371563747269223",
            "extra": "mean: 10.490935285486803 usec\nrounds: 70587"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 89831.27465602498,
            "unit": "iter/sec",
            "range": "stddev: 5.56527524150308e-7",
            "extra": "mean: 11.131980524924346 usec\nrounds: 51399"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 96962.7295048172,
            "unit": "iter/sec",
            "range": "stddev: 9.001379963626369e-7",
            "extra": "mean: 10.313241026804212 usec\nrounds: 56139"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2836.9426501703347,
            "unit": "iter/sec",
            "range": "stddev: 0.00016468918926684816",
            "extra": "mean: 352.4921449998146 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3254.896829539591,
            "unit": "iter/sec",
            "range": "stddev: 0.000014316381786914188",
            "extra": "mean: 307.229399999585 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2419.4408693409605,
            "unit": "iter/sec",
            "range": "stddev: 0.0019248487426454354",
            "extra": "mean: 413.31863600055385 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3151.825203998414,
            "unit": "iter/sec",
            "range": "stddev: 0.000013819042137985097",
            "extra": "mean: 317.2764780011903 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 10282.757081896296,
            "unit": "iter/sec",
            "range": "stddev: 0.000008058102759263077",
            "extra": "mean: 97.25018222598962 usec\nrounds: 7798"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 2450.2815742060225,
            "unit": "iter/sec",
            "range": "stddev: 0.00001838831007900075",
            "extra": "mean: 408.1163612080114 usec\nrounds: 2284"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1454.1093537949218,
            "unit": "iter/sec",
            "range": "stddev: 0.0001797531250617125",
            "extra": "mean: 687.7061875643732 usec\nrounds: 965"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 528040.2327264816,
            "unit": "iter/sec",
            "range": "stddev: 3.417380017439054e-7",
            "extra": "mean: 1.893795089886622 usec\nrounds: 109326"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 113070.72822478249,
            "unit": "iter/sec",
            "range": "stddev: 4.914801767014532e-7",
            "extra": "mean: 8.844021929460105 usec\nrounds: 22390"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 86869.75372451128,
            "unit": "iter/sec",
            "range": "stddev: 7.871690354576001e-7",
            "extra": "mean: 11.511486531564078 usec\nrounds: 22794"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 67720.43734757125,
            "unit": "iter/sec",
            "range": "stddev: 9.48706212470122e-7",
            "extra": "mean: 14.766590990361706 usec\nrounds: 22310"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 73604.77005275998,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016536806538256246",
            "extra": "mean: 13.58607600136783 usec\nrounds: 12855"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 85967.79027399923,
            "unit": "iter/sec",
            "range": "stddev: 7.564537906055505e-7",
            "extra": "mean: 11.632263628188753 usec\nrounds: 11190"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 42799.49399343162,
            "unit": "iter/sec",
            "range": "stddev: 0.00000196021361984382",
            "extra": "mean: 23.364762213157675 usec\nrounds: 10644"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 17232.406577793772,
            "unit": "iter/sec",
            "range": "stddev: 0.000012936841635187547",
            "extra": "mean: 58.03019998893433 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 51652.05693573795,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017480458673221064",
            "extra": "mean: 19.360313205805788 usec\nrounds: 15319"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 24833.003114048563,
            "unit": "iter/sec",
            "range": "stddev: 0.000003793515989108103",
            "extra": "mean: 40.268991849571286 usec\nrounds: 7116"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 31519.011812051758,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023932778392481156",
            "extra": "mean: 31.726882998839304 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 16947.06661683342,
            "unit": "iter/sec",
            "range": "stddev: 0.00000473017674586823",
            "extra": "mean: 59.00726200053441 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 12482.122480044689,
            "unit": "iter/sec",
            "range": "stddev: 0.000004141251283531974",
            "extra": "mean: 80.11458000021321 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 21303.470753104113,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035385838195881616",
            "extra": "mean: 46.940707999624465 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 51781.17907602828,
            "unit": "iter/sec",
            "range": "stddev: 0.000001610094528301098",
            "extra": "mean: 19.31203610739993 usec\nrounds: 16534"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 52393.000434639274,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035079617780984673",
            "extra": "mean: 19.08651903315804 usec\nrounds: 17995"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 7060.615819459423,
            "unit": "iter/sec",
            "range": "stddev: 0.00007070181737071073",
            "extra": "mean: 141.63070553193793 usec\nrounds: 3362"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 49505.74449483673,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020174201700054024",
            "extra": "mean: 20.199676021523047 usec\nrounds: 25548"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 35122.48957638781,
            "unit": "iter/sec",
            "range": "stddev: 0.00000291300654793924",
            "extra": "mean: 28.471785800522557 usec\nrounds: 15761"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 39487.90740549,
            "unit": "iter/sec",
            "range": "stddev: 0.000002479765545358534",
            "extra": "mean: 25.32420849074849 usec\nrounds: 11660"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 45374.932253326915,
            "unit": "iter/sec",
            "range": "stddev: 0.000004738314788521434",
            "extra": "mean: 22.03860039761667 usec\nrounds: 15090"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 40764.159015780926,
            "unit": "iter/sec",
            "range": "stddev: 0.000002325569006055171",
            "extra": "mean: 24.53135362397327 usec\nrounds: 14391"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 25471.645475726677,
            "unit": "iter/sec",
            "range": "stddev: 0.000003266225460476263",
            "extra": "mean: 39.2593403890202 usec\nrounds: 10535"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 16459.75317659701,
            "unit": "iter/sec",
            "range": "stddev: 0.000002385570951611361",
            "extra": "mean: 60.75425246484445 usec\nrounds: 10853"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 8220.177698095931,
            "unit": "iter/sec",
            "range": "stddev: 0.00000456265254988127",
            "extra": "mean: 121.65187137397693 usec\nrounds: 5551"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2498.089038047647,
            "unit": "iter/sec",
            "range": "stddev: 0.000011030723970992142",
            "extra": "mean: 400.30598780479767 usec\nrounds: 2132"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 911125.0927302772,
            "unit": "iter/sec",
            "range": "stddev: 7.299774780498415e-8",
            "extra": "mean: 1.0975441330492468 usec\nrounds: 173011"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3827.685529688651,
            "unit": "iter/sec",
            "range": "stddev: 0.000008909422076630388",
            "extra": "mean: 261.2544819169984 usec\nrounds: 2295"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 3961.408424248696,
            "unit": "iter/sec",
            "range": "stddev: 0.0000064871353680468075",
            "extra": "mean: 252.43547064694695 usec\nrounds: 3032"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1274.6292318604217,
            "unit": "iter/sec",
            "range": "stddev: 0.003353939844486333",
            "extra": "mean: 784.5418691209688 usec\nrounds: 1467"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 611.0122385774217,
            "unit": "iter/sec",
            "range": "stddev: 0.0028589901212169137",
            "extra": "mean: 1.6366284287991222 msec\nrounds: 625"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 606.8230619744614,
            "unit": "iter/sec",
            "range": "stddev: 0.0030848734711479047",
            "extra": "mean: 1.647926821940867 msec\nrounds: 629"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 11392.921257215048,
            "unit": "iter/sec",
            "range": "stddev: 0.00007981738476193101",
            "extra": "mean: 87.77380071565999 usec\nrounds: 2795"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 11763.441412120339,
            "unit": "iter/sec",
            "range": "stddev: 0.000010857699631236979",
            "extra": "mean: 85.00913677945132 usec\nrounds: 7004"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 188472.23171519328,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015433419946250876",
            "extra": "mean: 5.3058213981947935 usec\nrounds: 23404"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1574437.2587988458,
            "unit": "iter/sec",
            "range": "stddev: 7.00330580560006e-8",
            "extra": "mean: 635.1475706074882 nsec\nrounds: 68042"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 285903.21712478163,
            "unit": "iter/sec",
            "range": "stddev: 5.393732594670543e-7",
            "extra": "mean: 3.4976871196365478 usec\nrounds: 24805"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 3873.7409195227533,
            "unit": "iter/sec",
            "range": "stddev: 0.000050387267735606245",
            "extra": "mean: 258.14839473652785 usec\nrounds: 1634"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 3019.5339186861497,
            "unit": "iter/sec",
            "range": "stddev: 0.00005923761950928797",
            "extra": "mean: 331.1769388684718 usec\nrounds: 2650"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1690.3735799521605,
            "unit": "iter/sec",
            "range": "stddev: 0.00005857175914441095",
            "extra": "mean: 591.5852045133722 usec\nrounds: 1418"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 8.535097517505708,
            "unit": "iter/sec",
            "range": "stddev: 0.026505099223922746",
            "extra": "mean: 117.16327762500356 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 2.599512736674551,
            "unit": "iter/sec",
            "range": "stddev: 0.07888226734898965",
            "extra": "mean: 384.6874785000125 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 8.547279847481144,
            "unit": "iter/sec",
            "range": "stddev: 0.030048893287226177",
            "extra": "mean: 116.9962862857119 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 7.751229583916142,
            "unit": "iter/sec",
            "range": "stddev: 0.007702101936875279",
            "extra": "mean: 129.01178957142585 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 1.5147119306676822,
            "unit": "iter/sec",
            "range": "stddev: 0.209002980608988",
            "extra": "mean: 660.1915385714311 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_filesystem",
            "value": 10.541933319686494,
            "unit": "iter/sec",
            "range": "stddev: 0.004948319604589426",
            "extra": "mean: 94.85926060000338 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_filesystem",
            "value": 9.393006178565843,
            "unit": "iter/sec",
            "range": "stddev: 0.023274482965195235",
            "extra": "mean: 106.46218910000584 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 38567.05258414382,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022705754441677877",
            "extra": "mean: 25.92886759542348 usec\nrounds: 9116"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 29561.790642947188,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027485953261886427",
            "extra": "mean: 33.827450173035395 usec\nrounds: 7797"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 38933.141521682264,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015606291891988163",
            "extra": "mean: 25.685058048631646 usec\nrounds: 14195"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 26268.29238777137,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024060961312642575",
            "extra": "mean: 38.06870980564874 usec\nrounds: 9821"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 34211.65771683497,
            "unit": "iter/sec",
            "range": "stddev: 0.000002395102158903912",
            "extra": "mean: 29.229802550839775 usec\nrounds: 10428"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 25507.87602222708,
            "unit": "iter/sec",
            "range": "stddev: 0.000004925860342820538",
            "extra": "mean: 39.203577715707056 usec\nrounds: 10043"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 33789.27433091351,
            "unit": "iter/sec",
            "range": "stddev: 0.000002131400327216361",
            "extra": "mean: 29.595190183919062 usec\nrounds: 7335"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 23513.109745516584,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031892248687838315",
            "extra": "mean: 42.52946593721731 usec\nrounds: 9688"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 28237.649814778048,
            "unit": "iter/sec",
            "range": "stddev: 0.0000069066258237871065",
            "extra": "mean: 35.413712067378015 usec\nrounds: 7988"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 19605.733736486884,
            "unit": "iter/sec",
            "range": "stddev: 0.000003597698367077791",
            "extra": "mean: 51.005487141701245 usec\nrounds: 8205"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 9538.177601659856,
            "unit": "iter/sec",
            "range": "stddev: 0.000011684976182958429",
            "extra": "mean: 104.84183056373134 usec\nrounds: 3671"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 59736.218006197305,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013100631875397287",
            "extra": "mean: 16.740262999178412 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 30068.33058155079,
            "unit": "iter/sec",
            "range": "stddev: 0.000002148504601415086",
            "extra": "mean: 33.257583000420254 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 49953.011702150965,
            "unit": "iter/sec",
            "range": "stddev: 0.000002000009407135068",
            "extra": "mean: 20.018812998955582 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 25734.95294019097,
            "unit": "iter/sec",
            "range": "stddev: 0.000002252509278399505",
            "extra": "mean: 38.85765800015406 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 750.2456011213587,
            "unit": "iter/sec",
            "range": "stddev: 0.00003075250483851279",
            "extra": "mean: 1.3328968520513076 msec\nrounds: 561"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 974.2090346598168,
            "unit": "iter/sec",
            "range": "stddev: 0.00011373094438519282",
            "extra": "mean: 1.0264737488799713 msec\nrounds: 892"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6206.733148306341,
            "unit": "iter/sec",
            "range": "stddev: 0.0000073492597238969086",
            "extra": "mean: 161.11535265099556 usec\nrounds: 2433"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 6799.114054067448,
            "unit": "iter/sec",
            "range": "stddev: 0.000007508308072513068",
            "extra": "mean: 147.07798575636014 usec\nrounds: 3721"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 6864.1087314642255,
            "unit": "iter/sec",
            "range": "stddev: 0.000007907293512903837",
            "extra": "mean: 145.68533791082353 usec\nrounds: 2039"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4521.370202161886,
            "unit": "iter/sec",
            "range": "stddev: 0.00001841685815958075",
            "extra": "mean: 221.1718915477993 usec\nrounds: 2591"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 2102.4356837653495,
            "unit": "iter/sec",
            "range": "stddev: 0.00001931993067328547",
            "extra": "mean: 475.63880680005093 usec\nrounds: 1853"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 2106.633571801002,
            "unit": "iter/sec",
            "range": "stddev: 0.00002172097619873371",
            "extra": "mean: 474.69100150392103 usec\nrounds: 1996"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 2568.5378809555386,
            "unit": "iter/sec",
            "range": "stddev: 0.000019873379997715332",
            "extra": "mean: 389.3265532171102 usec\nrounds: 2471"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 2563.8147773744518,
            "unit": "iter/sec",
            "range": "stddev: 0.000022401181370278328",
            "extra": "mean: 390.04377727476816 usec\nrounds: 2429"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1831.8712437972868,
            "unit": "iter/sec",
            "range": "stddev: 0.000030013410130849626",
            "extra": "mean: 545.8898944923113 usec\nrounds: 1725"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1838.3704237167494,
            "unit": "iter/sec",
            "range": "stddev: 0.00002388005278055851",
            "extra": "mean: 543.9600132264078 usec\nrounds: 1739"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 2214.371369372643,
            "unit": "iter/sec",
            "range": "stddev: 0.00002721170924328515",
            "extra": "mean: 451.59543418559986 usec\nrounds: 2112"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 2218.614185920264,
            "unit": "iter/sec",
            "range": "stddev: 0.000019984839101547967",
            "extra": "mean: 450.73181553881017 usec\nrounds: 1789"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 585.6779181520577,
            "unit": "iter/sec",
            "range": "stddev: 0.0000853353133709157",
            "extra": "mean: 1.7074230887092676 msec\nrounds: 496"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 597.8181092228662,
            "unit": "iter/sec",
            "range": "stddev: 0.00001997795091653873",
            "extra": "mean: 1.6727495948557838 msec\nrounds: 311"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fd5e0cf5c500fbb726417f15e7c3966b5cf85e93",
          "message": "[BugFix] Fix vmap for tensorclass (#778)",
          "timestamp": "2024-05-15T11:34:03+01:00",
          "tree_id": "ef3c8d9d9b88a1e9219b0dfce98669f1b8267b50",
          "url": "https://github.com/pytorch/tensordict/commit/fd5e0cf5c500fbb726417f15e7c3966b5cf85e93"
        },
        "date": 1715769503976,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 61626.606598117636,
            "unit": "iter/sec",
            "range": "stddev: 8.128097768798822e-7",
            "extra": "mean: 16.22675748676619 usec\nrounds: 8148"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 60792.853545071564,
            "unit": "iter/sec",
            "range": "stddev: 8.391539405322503e-7",
            "extra": "mean: 16.449301878198632 usec\nrounds: 17570"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 53775.52706642351,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010906035113532362",
            "extra": "mean: 18.595819595869333 usec\nrounds: 30731"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 53843.48377056438,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010032438820185015",
            "extra": "mean: 18.572349520717466 usec\nrounds: 35995"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 371900.5797621653,
            "unit": "iter/sec",
            "range": "stddev: 1.7396444352638428e-7",
            "extra": "mean: 2.688890672446683 usec\nrounds: 73449"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 3795.8398898799987,
            "unit": "iter/sec",
            "range": "stddev: 0.000016788630169164137",
            "extra": "mean: 263.44630674915373 usec\nrounds: 3260"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 3749.09965554627,
            "unit": "iter/sec",
            "range": "stddev: 0.000005752171429266432",
            "extra": "mean: 266.7307065366053 usec\nrounds: 3595"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 12928.195128745634,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026490785006835235",
            "extra": "mean: 77.35031766162903 usec\nrounds: 10231"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 3777.5358290455424,
            "unit": "iter/sec",
            "range": "stddev: 0.000004272836316799168",
            "extra": "mean: 264.7228365938932 usec\nrounds: 3476"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 12793.200789989318,
            "unit": "iter/sec",
            "range": "stddev: 0.000006982435118745734",
            "extra": "mean: 78.16652114008093 usec\nrounds: 10241"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 3775.979451299351,
            "unit": "iter/sec",
            "range": "stddev: 0.000004784460347944796",
            "extra": "mean: 264.8319496696123 usec\nrounds: 3636"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 252741.160974731,
            "unit": "iter/sec",
            "range": "stddev: 2.4986423440686417e-7",
            "extra": "mean: 3.9566171024274897 usec\nrounds: 117995"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 7186.574545057593,
            "unit": "iter/sec",
            "range": "stddev: 0.0000059924647824100946",
            "extra": "mean: 139.14835137801884 usec\nrounds: 6059"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 7010.325604044719,
            "unit": "iter/sec",
            "range": "stddev: 0.000023861102545036197",
            "extra": "mean: 142.6467266260834 usec\nrounds: 6471"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 8541.848273419464,
            "unit": "iter/sec",
            "range": "stddev: 0.000005095643100424872",
            "extra": "mean: 117.07068165935486 usec\nrounds: 7737"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 7295.219165405336,
            "unit": "iter/sec",
            "range": "stddev: 0.000003468297253221064",
            "extra": "mean: 137.0760737034606 usec\nrounds: 6499"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 8547.304509692183,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027446491180852518",
            "extra": "mean: 116.9959487071104 usec\nrounds: 7233"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 7052.769070455183,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029537677180654104",
            "extra": "mean: 141.78828060443786 usec\nrounds: 6486"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 831271.48546828,
            "unit": "iter/sec",
            "range": "stddev: 6.386486780480481e-8",
            "extra": "mean: 1.2029764252489308 usec\nrounds: 193088"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 19800.06445017617,
            "unit": "iter/sec",
            "range": "stddev: 0.000001679462479738965",
            "extra": "mean: 50.5048861086461 usec\nrounds: 14268"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 19930.664904311885,
            "unit": "iter/sec",
            "range": "stddev: 0.000002708477562756334",
            "extra": "mean: 50.17394074914459 usec\nrounds: 17080"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 21841.98121611645,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025226263595485608",
            "extra": "mean: 45.78339254600834 usec\nrounds: 17789"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 19122.3069400364,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029097119304830016",
            "extra": "mean: 52.29494553851652 usec\nrounds: 15387"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 21935.710323731833,
            "unit": "iter/sec",
            "range": "stddev: 0.000001838809208025013",
            "extra": "mean: 45.58776466509583 usec\nrounds: 17456"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 19441.02147051647,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017068811222795616",
            "extra": "mean: 51.43762643935983 usec\nrounds: 17542"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 709185.9013658687,
            "unit": "iter/sec",
            "range": "stddev: 1.9309992611088958e-7",
            "extra": "mean: 1.4100675127269633 usec\nrounds: 144447"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 288037.4544555489,
            "unit": "iter/sec",
            "range": "stddev: 2.477508882335012e-7",
            "extra": "mean: 3.471770717770748 usec\nrounds: 102062"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 286665.04242960305,
            "unit": "iter/sec",
            "range": "stddev: 2.606984940344836e-7",
            "extra": "mean: 3.4883918580535402 usec\nrounds: 124147"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 287565.44598184805,
            "unit": "iter/sec",
            "range": "stddev: 2.8252361914871084e-7",
            "extra": "mean: 3.4774692647291245 usec\nrounds: 76313"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 268659.0972718759,
            "unit": "iter/sec",
            "range": "stddev: 0.000002963964932836176",
            "extra": "mean: 3.722189235929824 usec\nrounds: 49145"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 220401.85032601788,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014243508033768746",
            "extra": "mean: 4.537166990752584 usec\nrounds: 100311"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 238197.48742000104,
            "unit": "iter/sec",
            "range": "stddev: 3.3911297438815556e-7",
            "extra": "mean: 4.198197096163121 usec\nrounds: 102691"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 208297.6258187862,
            "unit": "iter/sec",
            "range": "stddev: 4.67007334243902e-7",
            "extra": "mean: 4.800822842167079 usec\nrounds: 55301"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 208413.58687226597,
            "unit": "iter/sec",
            "range": "stddev: 3.48092233397215e-7",
            "extra": "mean: 4.798151670470924 usec\nrounds: 67673"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 92935.87824686248,
            "unit": "iter/sec",
            "range": "stddev: 6.654021967228178e-7",
            "extra": "mean: 10.760107063751345 usec\nrounds: 69071"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 97878.34028330908,
            "unit": "iter/sec",
            "range": "stddev: 5.221776773587483e-7",
            "extra": "mean: 10.216764987079857 usec\nrounds: 74047"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 92758.76951844923,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020896751306706454",
            "extra": "mean: 10.780651847705949 usec\nrounds: 56234"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 98538.99312836886,
            "unit": "iter/sec",
            "range": "stddev: 5.714507213269682e-7",
            "extra": "mean: 10.148266876415903 usec\nrounds: 56262"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 88113.08547853092,
            "unit": "iter/sec",
            "range": "stddev: 7.727329685380135e-7",
            "extra": "mean: 11.349052125108633 usec\nrounds: 50513"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 94875.68682899832,
            "unit": "iter/sec",
            "range": "stddev: 6.657823576263737e-7",
            "extra": "mean: 10.54010815017736 usec\nrounds: 70245"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 89601.93441334844,
            "unit": "iter/sec",
            "range": "stddev: 5.416729229786883e-7",
            "extra": "mean: 11.160473337404031 usec\nrounds: 50539"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 96461.2303656009,
            "unit": "iter/sec",
            "range": "stddev: 5.542270312512178e-7",
            "extra": "mean: 10.366859267810154 usec\nrounds: 56874"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2818.294225339766,
            "unit": "iter/sec",
            "range": "stddev: 0.00014898446910153752",
            "extra": "mean: 354.8245569993469 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3175.580668738941,
            "unit": "iter/sec",
            "range": "stddev: 0.00001987702319626355",
            "extra": "mean: 314.9030380000113 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2451.1172702370245,
            "unit": "iter/sec",
            "range": "stddev: 0.001692971734553973",
            "extra": "mean: 407.9772159996651 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3113.9520290517908,
            "unit": "iter/sec",
            "range": "stddev: 0.000015480294870807985",
            "extra": "mean: 321.13532600067174 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 10402.162843719723,
            "unit": "iter/sec",
            "range": "stddev: 0.000008540950627619303",
            "extra": "mean: 96.13385360562272 usec\nrounds: 7794"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 2425.609424870504,
            "unit": "iter/sec",
            "range": "stddev: 0.00001744866838855625",
            "extra": "mean: 412.2675273878386 usec\nrounds: 2209"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1485.913445125376,
            "unit": "iter/sec",
            "range": "stddev: 0.00007144956380734872",
            "extra": "mean: 672.9867094752774 usec\nrounds: 950"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 518097.90753727895,
            "unit": "iter/sec",
            "range": "stddev: 1.7655107809106856e-7",
            "extra": "mean: 1.930137113954753 usec\nrounds: 116064"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 111705.69436876435,
            "unit": "iter/sec",
            "range": "stddev: 6.164481435948103e-7",
            "extra": "mean: 8.952095107155294 usec\nrounds: 23647"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 85265.58476919947,
            "unit": "iter/sec",
            "range": "stddev: 6.790266549613073e-7",
            "extra": "mean: 11.728061241904841 usec\nrounds: 24346"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 65707.79351359997,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013755385227832736",
            "extra": "mean: 15.218894845297514 usec\nrounds: 23261"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 73379.98567722054,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021445418202785536",
            "extra": "mean: 13.627694129005965 usec\nrounds: 14274"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 85791.3585519636,
            "unit": "iter/sec",
            "range": "stddev: 7.371826800612352e-7",
            "extra": "mean: 11.656185621472618 usec\nrounds: 10947"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 43595.39493576785,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017339909856315508",
            "extra": "mean: 22.938202566426344 usec\nrounds: 10209"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 16653.288522241317,
            "unit": "iter/sec",
            "range": "stddev: 0.000012313359541598471",
            "extra": "mean: 60.04820000953259 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 51680.18494074072,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013130980953876556",
            "extra": "mean: 19.349775956619617 usec\nrounds: 15680"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 23706.77770036483,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030976918440657044",
            "extra": "mean: 42.18202965578957 usec\nrounds: 7351"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 30004.113864627514,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029698789610613328",
            "extra": "mean: 33.328762999360606 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 16359.248575206826,
            "unit": "iter/sec",
            "range": "stddev: 0.000004667650958523507",
            "extra": "mean: 61.12750200003347 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 12092.769487722717,
            "unit": "iter/sec",
            "range": "stddev: 0.000006219450828469566",
            "extra": "mean: 82.69404299943517 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 21032.182098787947,
            "unit": "iter/sec",
            "range": "stddev: 0.00000295027516689628",
            "extra": "mean: 47.54618400045274 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 51967.20680342794,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031202953326695978",
            "extra": "mean: 19.242904545218632 usec\nrounds: 15819"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 53810.35638843133,
            "unit": "iter/sec",
            "range": "stddev: 0.000001940520853041692",
            "extra": "mean: 18.58378325505738 usec\nrounds: 17629"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 7089.606711549156,
            "unit": "iter/sec",
            "range": "stddev: 0.00008604083830988018",
            "extra": "mean: 141.05154780602618 usec\nrounds: 3713"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 51182.364067896364,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017784879826288136",
            "extra": "mean: 19.537979892320763 usec\nrounds: 25960"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 36614.99050268317,
            "unit": "iter/sec",
            "range": "stddev: 0.000002180646393989111",
            "extra": "mean: 27.311218336290963 usec\nrounds: 16012"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 40171.271841062386,
            "unit": "iter/sec",
            "range": "stddev: 0.000002165541075400976",
            "extra": "mean: 24.893411489596335 usec\nrounds: 11541"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 49556.57385370354,
            "unit": "iter/sec",
            "range": "stddev: 0.000002127648459842423",
            "extra": "mean: 20.17895754763253 usec\nrounds: 17290"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 41055.90141841252,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019064646716431946",
            "extra": "mean: 24.357034322756967 usec\nrounds: 15704"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 25302.074932099746,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025242009844057344",
            "extra": "mean: 39.522450339886525 usec\nrounds: 11025"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 16222.327152883594,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022162797713320825",
            "extra": "mean: 61.643436886442366 usec\nrounds: 11408"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 8046.874961475332,
            "unit": "iter/sec",
            "range": "stddev: 0.000013890130439362615",
            "extra": "mean: 124.27184525514956 usec\nrounds: 5564"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2476.4687512596797,
            "unit": "iter/sec",
            "range": "stddev: 0.000009052490999536048",
            "extra": "mean: 403.8007745873395 usec\nrounds: 2125"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 926392.9302545069,
            "unit": "iter/sec",
            "range": "stddev: 8.058042989274281e-8",
            "extra": "mean: 1.0794555607470495 usec\nrounds: 169177"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3804.2573325669546,
            "unit": "iter/sec",
            "range": "stddev: 0.000023752804833547276",
            "extra": "mean: 262.8633955540651 usec\nrounds: 2384"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 3901.115114733097,
            "unit": "iter/sec",
            "range": "stddev: 0.000018883803302063044",
            "extra": "mean: 256.33696278875817 usec\nrounds: 3413"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1292.2395256129464,
            "unit": "iter/sec",
            "range": "stddev: 0.00301283685037397",
            "extra": "mean: 773.8503428965084 usec\nrounds: 1464"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 608.4755834397162,
            "unit": "iter/sec",
            "range": "stddev: 0.0028348499344899793",
            "extra": "mean: 1.6434513186987618 msec\nrounds: 615"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 614.0957557143515,
            "unit": "iter/sec",
            "range": "stddev: 0.0025015464165496883",
            "extra": "mean: 1.6284105380874072 msec\nrounds: 617"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 11646.304921978759,
            "unit": "iter/sec",
            "range": "stddev: 0.00006680285377872245",
            "extra": "mean: 85.86414375196486 usec\nrounds: 2713"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 11587.62187137571,
            "unit": "iter/sec",
            "range": "stddev: 0.000008247187444040823",
            "extra": "mean: 86.29898447672402 usec\nrounds: 7215"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 179139.53043284093,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016808864712301946",
            "extra": "mean: 5.582240824142933 usec\nrounds: 21933"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1570732.5437253187,
            "unit": "iter/sec",
            "range": "stddev: 1.701902755111922e-7",
            "extra": "mean: 636.6456237217141 nsec\nrounds: 74102"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 273593.1161756355,
            "unit": "iter/sec",
            "range": "stddev: 4.369020607987216e-7",
            "extra": "mean: 3.6550627222581187 usec\nrounds: 26434"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 3930.811891725948,
            "unit": "iter/sec",
            "range": "stddev: 0.00005221181337210488",
            "extra": "mean: 254.4003700876457 usec\nrounds: 1705"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 3020.5774812146587,
            "unit": "iter/sec",
            "range": "stddev: 0.000056795440611807525",
            "extra": "mean: 331.0625223882262 usec\nrounds: 2680"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1681.3127410465913,
            "unit": "iter/sec",
            "range": "stddev: 0.00007655653967542576",
            "extra": "mean: 594.7733432255533 usec\nrounds: 1550"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 8.635874575010101,
            "unit": "iter/sec",
            "range": "stddev: 0.023341238764105755",
            "extra": "mean: 115.79603100000213 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 2.6387011846510453,
            "unit": "iter/sec",
            "range": "stddev: 0.11805552758288301",
            "extra": "mean: 378.9743248750028 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.113134359472514,
            "unit": "iter/sec",
            "range": "stddev: 0.022415273196707115",
            "extra": "mean: 109.73173011111864 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 7.390562501065176,
            "unit": "iter/sec",
            "range": "stddev: 0.02481395986352324",
            "extra": "mean: 135.3076981428509 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 1.5505498369986541,
            "unit": "iter/sec",
            "range": "stddev: 0.2622632810277093",
            "extra": "mean: 644.9325111249991 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_filesystem",
            "value": 10.166686982942139,
            "unit": "iter/sec",
            "range": "stddev: 0.018448532376774532",
            "extra": "mean: 98.36045918181794 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_filesystem",
            "value": 10.696336039971243,
            "unit": "iter/sec",
            "range": "stddev: 0.003322853097004799",
            "extra": "mean: 93.48995733334202 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 39128.53628384916,
            "unit": "iter/sec",
            "range": "stddev: 0.000001543530532347198",
            "extra": "mean: 25.55679549947192 usec\nrounds: 9022"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 29908.169907574455,
            "unit": "iter/sec",
            "range": "stddev: 0.000001532817231629727",
            "extra": "mean: 33.43568005298589 usec\nrounds: 7520"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 39529.127946142784,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015236127822708014",
            "extra": "mean: 25.29780068415547 usec\nrounds: 14033"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 26605.154751357815,
            "unit": "iter/sec",
            "range": "stddev: 0.000002080123701399513",
            "extra": "mean: 37.586701124110704 usec\nrounds: 9519"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 33964.976777622,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015403726836227761",
            "extra": "mean: 29.44209285191842 usec\nrounds: 10016"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 26138.06073971334,
            "unit": "iter/sec",
            "range": "stddev: 0.000004780108542737509",
            "extra": "mean: 38.25838534687586 usec\nrounds: 11042"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 33712.228425311245,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020423805977124783",
            "extra": "mean: 29.662827012918463 usec\nrounds: 8307"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 24250.086019615737,
            "unit": "iter/sec",
            "range": "stddev: 0.000004759336458879799",
            "extra": "mean: 41.236967126265306 usec\nrounds: 10008"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 29109.985266153017,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021604498739329116",
            "extra": "mean: 34.35247358791101 usec\nrounds: 8159"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 19875.15517106964,
            "unit": "iter/sec",
            "range": "stddev: 0.000002983022795482059",
            "extra": "mean: 50.3140725892598 usec\nrounds: 8431"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 9775.447742106455,
            "unit": "iter/sec",
            "range": "stddev: 0.000008640000646228056",
            "extra": "mean: 102.29710458096272 usec\nrounds: 4016"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 58331.71414069373,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017620954056795546",
            "extra": "mean: 17.14333300043336 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 29594.068519391025,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023894611301272743",
            "extra": "mean: 33.79055500073491 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 51773.23592188656,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014354245545857398",
            "extra": "mean: 19.314998998879673 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 25830.574370122762,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026517415983535053",
            "extra": "mean: 38.71381199934376 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 774.6993310355075,
            "unit": "iter/sec",
            "range": "stddev: 0.000022134107096524484",
            "extra": "mean: 1.290823368420033 msec\nrounds: 589"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 995.7673116694018,
            "unit": "iter/sec",
            "range": "stddev: 0.00009810673719466248",
            "extra": "mean: 1.004250680134802 msec\nrounds: 891"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6212.206357014955,
            "unit": "iter/sec",
            "range": "stddev: 0.000006538923239739889",
            "extra": "mean: 160.9734034141958 usec\nrounds: 2402"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 6645.219034989447,
            "unit": "iter/sec",
            "range": "stddev: 0.000007804813280563086",
            "extra": "mean: 150.48412922653768 usec\nrounds: 3815"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 6826.082241727173,
            "unit": "iter/sec",
            "range": "stddev: 0.000006959027864640973",
            "extra": "mean: 146.49691647239433 usec\nrounds: 2143"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4485.350747952294,
            "unit": "iter/sec",
            "range": "stddev: 0.000019181508329554537",
            "extra": "mean: 222.94800478123852 usec\nrounds: 1464"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 2071.8702468047504,
            "unit": "iter/sec",
            "range": "stddev: 0.000023374619700116577",
            "extra": "mean: 482.65570758700034 usec\nrounds: 1898"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 2091.5416292089562,
            "unit": "iter/sec",
            "range": "stddev: 0.00001692756030220457",
            "extra": "mean: 478.11623064763523 usec\nrounds: 1899"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 2529.957263488964,
            "unit": "iter/sec",
            "range": "stddev: 0.000016754747313512985",
            "extra": "mean: 395.26359374977716 usec\nrounds: 2272"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 2524.9224758692967,
            "unit": "iter/sec",
            "range": "stddev: 0.000015485279239543003",
            "extra": "mean: 396.05176378958464 usec\nrounds: 2375"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1813.624386554511,
            "unit": "iter/sec",
            "range": "stddev: 0.000030310032977272744",
            "extra": "mean: 551.3820873900912 usec\nrounds: 1705"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1813.130320781799,
            "unit": "iter/sec",
            "range": "stddev: 0.000021931032899724507",
            "extra": "mean: 551.5323352867501 usec\nrounds: 1706"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 2049.917102504019,
            "unit": "iter/sec",
            "range": "stddev: 0.0015037194032991906",
            "extra": "mean: 487.8246046039997 usec\nrounds: 1998"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 2211.7734177062102,
            "unit": "iter/sec",
            "range": "stddev: 0.00001910353508018311",
            "extra": "mean: 452.12587871549783 usec\nrounds: 2086"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 582.1454169009646,
            "unit": "iter/sec",
            "range": "stddev: 0.00002146084680890629",
            "extra": "mean: 1.7177838577231663 msec\nrounds: 492"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 591.6365889487888,
            "unit": "iter/sec",
            "range": "stddev: 0.000020125203920840457",
            "extra": "mean: 1.6902267687277173 msec\nrounds: 307"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1df73e74b693816cc5bc3897643c410bd035c95c",
          "message": "[Feature] flatten and unflatten as decorators (#779)",
          "timestamp": "2024-05-15T13:30:05+01:00",
          "tree_id": "fcd427d4ec8585fee4a2ee3b2dfc568920159ff5",
          "url": "https://github.com/pytorch/tensordict/commit/1df73e74b693816cc5bc3897643c410bd035c95c"
        },
        "date": 1715776467976,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 57710.558106505676,
            "unit": "iter/sec",
            "range": "stddev: 7.81162721773396e-7",
            "extra": "mean: 17.327851831799745 usec\nrounds: 7316"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 57365.10818595041,
            "unit": "iter/sec",
            "range": "stddev: 7.999932673773446e-7",
            "extra": "mean: 17.43219932155406 usec\nrounds: 15618"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 50743.57340570203,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011422162800806147",
            "extra": "mean: 19.706929033256266 usec\nrounds: 28774"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 50849.14434581806,
            "unit": "iter/sec",
            "range": "stddev: 9.346188269310385e-7",
            "extra": "mean: 19.666014302996665 usec\nrounds: 32651"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 387424.274252882,
            "unit": "iter/sec",
            "range": "stddev: 1.748532432074095e-7",
            "extra": "mean: 2.5811495728511678 usec\nrounds: 81686"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 3629.8733294423255,
            "unit": "iter/sec",
            "range": "stddev: 0.000023457600892410898",
            "extra": "mean: 275.49170707663086 usec\nrounds: 3250"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 3716.506769318583,
            "unit": "iter/sec",
            "range": "stddev: 0.000006816122255355252",
            "extra": "mean: 269.0698718095834 usec\nrounds: 3409"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 12973.25295448162,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022779545621560986",
            "extra": "mean: 77.08166976383121 usec\nrounds: 10547"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 3673.1277139725576,
            "unit": "iter/sec",
            "range": "stddev: 0.000006935777011704124",
            "extra": "mean: 272.2475442920227 usec\nrounds: 3206"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 12655.47319260101,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020483564084675105",
            "extra": "mean: 79.01719554703395 usec\nrounds: 10959"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 3667.055576039389,
            "unit": "iter/sec",
            "range": "stddev: 0.000006270507593030776",
            "extra": "mean: 272.6983486517137 usec\nrounds: 3634"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 260649.71410702562,
            "unit": "iter/sec",
            "range": "stddev: 2.709446813554037e-7",
            "extra": "mean: 3.836566648177443 usec\nrounds: 110303"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 7337.129477422928,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035096336722323403",
            "extra": "mean: 136.29308342957538 usec\nrounds: 6053"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 7080.212917901,
            "unit": "iter/sec",
            "range": "stddev: 0.000022866420633127207",
            "extra": "mean: 141.23869036080626 usec\nrounds: 6598"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 8659.07268048521,
            "unit": "iter/sec",
            "range": "stddev: 0.000002232561577208618",
            "extra": "mean: 115.48580741834877 usec\nrounds: 7576"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 7329.371528988617,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026705807999921697",
            "extra": "mean: 136.43734610053127 usec\nrounds: 6501"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 8504.809752002351,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027044943966119742",
            "extra": "mean: 117.58052550964617 usec\nrounds: 7448"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 7093.595769920447,
            "unit": "iter/sec",
            "range": "stddev: 0.0000043042684471528226",
            "extra": "mean: 140.97222796940045 usec\nrounds: 3983"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 890241.1320882938,
            "unit": "iter/sec",
            "range": "stddev: 5.309269874469239e-8",
            "extra": "mean: 1.1232911668035777 usec\nrounds: 161787"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 19711.822369148547,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015100453191710228",
            "extra": "mean: 50.73097663284164 usec\nrounds: 15021"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 19364.89213969197,
            "unit": "iter/sec",
            "range": "stddev: 0.000007648018313813323",
            "extra": "mean: 51.639843526435804 usec\nrounds: 17888"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 21514.69256080099,
            "unit": "iter/sec",
            "range": "stddev: 0.00000142551227380275",
            "extra": "mean: 46.47986473308777 usec\nrounds: 17824"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 19384.661711144305,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014744942604963117",
            "extra": "mean: 51.58717830113572 usec\nrounds: 16102"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 21753.30116751001,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015792098523107959",
            "extra": "mean: 45.97003426282563 usec\nrounds: 17570"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 19529.32194819633,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015148863519861243",
            "extra": "mean: 51.20505477110827 usec\nrounds: 17564"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 724031.1912370333,
            "unit": "iter/sec",
            "range": "stddev: 1.5494454842805963e-7",
            "extra": "mean: 1.3811559669017353 usec\nrounds: 161265"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 289570.6867997506,
            "unit": "iter/sec",
            "range": "stddev: 2.2993444321315464e-7",
            "extra": "mean: 3.453388224656658 usec\nrounds: 110060"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 287984.1963244459,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010209678216747353",
            "extra": "mean: 3.4724127669609683 usec\nrounds: 119261"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 287806.84025001246,
            "unit": "iter/sec",
            "range": "stddev: 2.283284744527001e-7",
            "extra": "mean: 3.474552582319859 usec\nrounds: 79599"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 288693.40300589707,
            "unit": "iter/sec",
            "range": "stddev: 2.4886045197351404e-7",
            "extra": "mean: 3.4638824080769632 usec\nrounds: 59698"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 235682.11140469438,
            "unit": "iter/sec",
            "range": "stddev: 4.893641538254623e-7",
            "extra": "mean: 4.2430033999605525 usec\nrounds: 99711"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 231706.5789574856,
            "unit": "iter/sec",
            "range": "stddev: 8.626699310477795e-7",
            "extra": "mean: 4.315803221899383 usec\nrounds: 97953"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 176258.48845571827,
            "unit": "iter/sec",
            "range": "stddev: 3.8876193917935657e-7",
            "extra": "mean: 5.673485621949106 usec\nrounds: 56266"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 173832.73401344978,
            "unit": "iter/sec",
            "range": "stddev: 3.5954403248480965e-7",
            "extra": "mean: 5.752656458378133 usec\nrounds: 62074"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 94513.42093608945,
            "unit": "iter/sec",
            "range": "stddev: 5.328613447533132e-7",
            "extra": "mean: 10.580507933113607 usec\nrounds: 70592"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 100557.13277254572,
            "unit": "iter/sec",
            "range": "stddev: 4.5143830064235315e-7",
            "extra": "mean: 9.944595399929916 usec\nrounds: 76782"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 95080.4260987883,
            "unit": "iter/sec",
            "range": "stddev: 5.258653753803304e-7",
            "extra": "mean: 10.517411848375636 usec\nrounds: 58337"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 100520.60738459081,
            "unit": "iter/sec",
            "range": "stddev: 4.0903376508788703e-7",
            "extra": "mean: 9.94820888988474 usec\nrounds: 58921"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 90872.651054427,
            "unit": "iter/sec",
            "range": "stddev: 4.842409212340415e-7",
            "extra": "mean: 11.004410990508715 usec\nrounds: 50006"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 97821.70204674511,
            "unit": "iter/sec",
            "range": "stddev: 5.389672274160249e-7",
            "extra": "mean: 10.222680438765416 usec\nrounds: 69464"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 91326.3221756248,
            "unit": "iter/sec",
            "range": "stddev: 4.740175609060453e-7",
            "extra": "mean: 10.949745661244885 usec\nrounds: 54337"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 98397.93390536502,
            "unit": "iter/sec",
            "range": "stddev: 4.5612466440412595e-7",
            "extra": "mean: 10.162815013593253 usec\nrounds: 58507"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2801.8233212683654,
            "unit": "iter/sec",
            "range": "stddev: 0.0001468899375973975",
            "extra": "mean: 356.91044200007127 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3177.8474036983753,
            "unit": "iter/sec",
            "range": "stddev: 0.000010763011022891965",
            "extra": "mean: 314.67842000097335 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2479.9120797749333,
            "unit": "iter/sec",
            "range": "stddev: 0.0015916157205162055",
            "extra": "mean: 403.2401020002112 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3109.762726281832,
            "unit": "iter/sec",
            "range": "stddev: 0.000007270367654834998",
            "extra": "mean: 321.5679420003994 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 10334.583926212184,
            "unit": "iter/sec",
            "range": "stddev: 0.000011964017167350745",
            "extra": "mean: 96.76248285754824 usec\nrounds: 7671"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 2431.424931251062,
            "unit": "iter/sec",
            "range": "stddev: 0.000010201132082210464",
            "extra": "mean: 411.2814618074437 usec\nrounds: 2291"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1334.8605263952268,
            "unit": "iter/sec",
            "range": "stddev: 0.0001279789224738097",
            "extra": "mean: 749.1419367238964 usec\nrounds: 885"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 479141.1117718046,
            "unit": "iter/sec",
            "range": "stddev: 1.8704095431275093e-7",
            "extra": "mean: 2.087067829145622 usec\nrounds: 116469"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 85431.65389838701,
            "unit": "iter/sec",
            "range": "stddev: 6.574986797635176e-7",
            "extra": "mean: 11.705263264473455 usec\nrounds: 20280"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 69176.6978518797,
            "unit": "iter/sec",
            "range": "stddev: 6.557784637868954e-7",
            "extra": "mean: 14.455734821878718 usec\nrounds: 21314"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 56591.29068815956,
            "unit": "iter/sec",
            "range": "stddev: 9.397107694358069e-7",
            "extra": "mean: 17.67056357683016 usec\nrounds: 21360"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 72296.67967474865,
            "unit": "iter/sec",
            "range": "stddev: 0.000001474379926038364",
            "extra": "mean: 13.831893864266549 usec\nrounds: 13690"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 85086.85565537836,
            "unit": "iter/sec",
            "range": "stddev: 6.851273076307787e-7",
            "extra": "mean: 11.752696609805792 usec\nrounds: 12034"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 43307.99543708289,
            "unit": "iter/sec",
            "range": "stddev: 0.000002548050823050449",
            "extra": "mean: 23.09042452571564 usec\nrounds: 12077"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 16467.13982740517,
            "unit": "iter/sec",
            "range": "stddev: 0.00001291467370551764",
            "extra": "mean: 60.726999981852714 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 51812.948727789146,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012869390337162106",
            "extra": "mean: 19.30019473035056 usec\nrounds: 15447"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 24061.28966354591,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034307568726853704",
            "extra": "mean: 41.56053204060177 usec\nrounds: 8973"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 28368.56657307815,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031349430787196786",
            "extra": "mean: 35.25028299981159 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 15902.513773891256,
            "unit": "iter/sec",
            "range": "stddev: 0.000004819410235868119",
            "extra": "mean: 62.88314000028094 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 11773.57415392679,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036496195187663455",
            "extra": "mean: 84.93597500012129 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 19612.244740014787,
            "unit": "iter/sec",
            "range": "stddev: 0.000003015324676938066",
            "extra": "mean: 50.9885540006394 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 47080.86689610306,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017887652137840207",
            "extra": "mean: 21.240050702693647 usec\nrounds: 15719"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 48469.779401743195,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016311190747898172",
            "extra": "mean: 20.631412239603375 usec\nrounds: 17844"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 7012.91076872495,
            "unit": "iter/sec",
            "range": "stddev: 0.00006372033894535509",
            "extra": "mean: 142.5941428571484 usec\nrounds: 3612"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 43591.86889461521,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018447526832535857",
            "extra": "mean: 22.940057982316226 usec\nrounds: 23766"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 30673.919004119285,
            "unit": "iter/sec",
            "range": "stddev: 0.000006293528135143511",
            "extra": "mean: 32.60098586899532 usec\nrounds: 15781"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 39484.76036456443,
            "unit": "iter/sec",
            "range": "stddev: 0.000008192407196958224",
            "extra": "mean: 25.326226897845107 usec\nrounds: 12380"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 44607.05334322075,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017558073600961908",
            "extra": "mean: 22.41797933402335 usec\nrounds: 16162"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 37085.25725015966,
            "unit": "iter/sec",
            "range": "stddev: 0.000002219127572803518",
            "extra": "mean: 26.964893171819504 usec\nrounds: 14865"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 23735.44860820116,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029902736093650617",
            "extra": "mean: 42.13107645475368 usec\nrounds: 9522"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 15289.781844281557,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019171462298238803",
            "extra": "mean: 65.40315683928507 usec\nrounds: 10769"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 8001.2690151926745,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032754334225979205",
            "extra": "mean: 124.98017478242726 usec\nrounds: 5298"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2510.8264114896656,
            "unit": "iter/sec",
            "range": "stddev: 0.000019673288356707845",
            "extra": "mean: 398.2752433318172 usec\nrounds: 2137"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 757489.2145840521,
            "unit": "iter/sec",
            "range": "stddev: 9.088258709612768e-8",
            "extra": "mean: 1.3201508097367616 usec\nrounds: 178891"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3764.74307640444,
            "unit": "iter/sec",
            "range": "stddev: 0.000054574268389579085",
            "extra": "mean: 265.6223757386018 usec\nrounds: 676"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 3837.4094287687712,
            "unit": "iter/sec",
            "range": "stddev: 0.000006672041173128821",
            "extra": "mean: 260.59246962366717 usec\nrounds: 3292"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1257.3083649817843,
            "unit": "iter/sec",
            "range": "stddev: 0.003121619125703232",
            "extra": "mean: 795.3498344970353 usec\nrounds: 1432"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 614.2105373040954,
            "unit": "iter/sec",
            "range": "stddev: 0.0026622076095028017",
            "extra": "mean: 1.6281062262285813 msec\nrounds: 610"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 614.4801450980904,
            "unit": "iter/sec",
            "range": "stddev: 0.002762636941707712",
            "extra": "mean: 1.6273918823534461 msec\nrounds: 629"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 11814.307338232267,
            "unit": "iter/sec",
            "range": "stddev: 0.000011992194892283813",
            "extra": "mean: 84.64313407218562 usec\nrounds: 2797"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 11904.129209990855,
            "unit": "iter/sec",
            "range": "stddev: 0.00000861689731847007",
            "extra": "mean: 84.00446453157812 usec\nrounds: 7119"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 186554.186490743,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017442081559914766",
            "extra": "mean: 5.360372869732521 usec\nrounds: 22005"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1588378.876203842,
            "unit": "iter/sec",
            "range": "stddev: 9.6889070852317e-8",
            "extra": "mean: 629.5727140302681 nsec\nrounds: 68790"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 280158.17932156014,
            "unit": "iter/sec",
            "range": "stddev: 5.018183611280844e-7",
            "extra": "mean: 3.569412117189052 usec\nrounds: 21045"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 4023.310329359342,
            "unit": "iter/sec",
            "range": "stddev: 0.0000510640327194061",
            "extra": "mean: 248.5515454034679 usec\nrounds: 1773"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 3089.9712751985217,
            "unit": "iter/sec",
            "range": "stddev: 0.00005092301888760722",
            "extra": "mean: 323.62760392837407 usec\nrounds: 2444"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1635.8183371572586,
            "unit": "iter/sec",
            "range": "stddev: 0.00006557050719597258",
            "extra": "mean: 611.314824687569 usec\nrounds: 1523"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 8.773606218282035,
            "unit": "iter/sec",
            "range": "stddev: 0.023701347660056632",
            "extra": "mean: 113.97821774998818 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 2.6381821863394657,
            "unit": "iter/sec",
            "range": "stddev: 0.08567022695008292",
            "extra": "mean: 379.04887887501104 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.159777824214963,
            "unit": "iter/sec",
            "range": "stddev: 0.02270060548489007",
            "extra": "mean: 109.17295366666873 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 7.505810167231033,
            "unit": "iter/sec",
            "range": "stddev: 0.023185974598870097",
            "extra": "mean: 133.2301214285719 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 1.5314750670115531,
            "unit": "iter/sec",
            "range": "stddev: 0.18814273746419088",
            "extra": "mean: 652.9652500000225 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_filesystem",
            "value": 10.098252953399175,
            "unit": "iter/sec",
            "range": "stddev: 0.020643775971428396",
            "extra": "mean: 99.02703018182861 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_filesystem",
            "value": 10.963378970405202,
            "unit": "iter/sec",
            "range": "stddev: 0.002753104759855153",
            "extra": "mean: 91.21275500002537 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 39315.98938190253,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016617484928135093",
            "extra": "mean: 25.43494429928573 usec\nrounds: 9228"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 29737.87388109759,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018272533023155357",
            "extra": "mean: 33.62715182660164 usec\nrounds: 8075"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 40110.22213803145,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018155174037355585",
            "extra": "mean: 24.931300468960167 usec\nrounds: 13662"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 26675.75820238113,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018998709744904301",
            "extra": "mean: 37.48721938522962 usec\nrounds: 9791"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 34583.47579734852,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013643847739912854",
            "extra": "mean: 28.915543534715184 usec\nrounds: 10084"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 25420.470469073778,
            "unit": "iter/sec",
            "range": "stddev: 0.000004037184272309293",
            "extra": "mean: 39.33837500043862 usec\nrounds: 11208"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 35155.110689151305,
            "unit": "iter/sec",
            "range": "stddev: 0.000001548666103751934",
            "extra": "mean: 28.44536627525383 usec\nrounds: 8267"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 23774.69681964212,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023211980476742007",
            "extra": "mean: 42.06152480454861 usec\nrounds: 10240"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 29105.708405189864,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022220078883176923",
            "extra": "mean: 34.35752142083884 usec\nrounds: 8496"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 17929.959124502006,
            "unit": "iter/sec",
            "range": "stddev: 0.000003874041938028603",
            "extra": "mean: 55.77257555670944 usec\nrounds: 7716"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 9748.17728368334,
            "unit": "iter/sec",
            "range": "stddev: 0.000007774413172171554",
            "extra": "mean: 102.58328002239111 usec\nrounds: 3664"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 56392.81339772475,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016961413483662475",
            "extra": "mean: 17.73275599759927 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 27936.60123722689,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017433568249371774",
            "extra": "mean: 35.79533499828358 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 48695.74770192665,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013761417703487132",
            "extra": "mean: 20.535674000143445 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 24568.34824843748,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017482529158876813",
            "extra": "mean: 40.70277699941016 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 767.8970929283478,
            "unit": "iter/sec",
            "range": "stddev: 0.000018918899449298876",
            "extra": "mean: 1.3022578275254777 msec\nrounds: 603"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 992.0659069284749,
            "unit": "iter/sec",
            "range": "stddev: 0.00008878890022968849",
            "extra": "mean: 1.0079975463485988 msec\nrounds: 917"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6167.406884459747,
            "unit": "iter/sec",
            "range": "stddev: 0.000008697636232049985",
            "extra": "mean: 162.14269931172186 usec\nrounds: 2471"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 6585.746890217902,
            "unit": "iter/sec",
            "range": "stddev: 0.0000065380806390949176",
            "extra": "mean: 151.84306604393555 usec\nrounds: 4073"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 6853.79255669589,
            "unit": "iter/sec",
            "range": "stddev: 0.000007705958825355662",
            "extra": "mean: 145.90462021250391 usec\nrounds: 2167"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4499.80918931953,
            "unit": "iter/sec",
            "range": "stddev: 0.000019275407439643318",
            "extra": "mean: 222.231645371439 usec\nrounds: 2755"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 2038.819040086156,
            "unit": "iter/sec",
            "range": "stddev: 0.00001900171723875454",
            "extra": "mean: 490.48001825495123 usec\nrounds: 1698"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 2061.242134641452,
            "unit": "iter/sec",
            "range": "stddev: 0.000012925738504363039",
            "extra": "mean: 485.1443618359507 usec\nrounds: 1918"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 2509.478339392064,
            "unit": "iter/sec",
            "range": "stddev: 0.000016152998676378194",
            "extra": "mean: 398.4891936713253 usec\nrounds: 2370"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 2522.408615464998,
            "unit": "iter/sec",
            "range": "stddev: 0.000014918680329689432",
            "extra": "mean: 396.44647337031597 usec\nrounds: 2347"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1784.5418168509143,
            "unit": "iter/sec",
            "range": "stddev: 0.00002638934649457478",
            "extra": "mean: 560.3679278105383 usec\nrounds: 1690"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1771.5817048109757,
            "unit": "iter/sec",
            "range": "stddev: 0.000016784730427054026",
            "extra": "mean: 564.4673329400283 usec\nrounds: 1691"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 2158.2790354102394,
            "unit": "iter/sec",
            "range": "stddev: 0.000019861025482727414",
            "extra": "mean: 463.3321195236106 usec\nrounds: 2008"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 2161.113906033772,
            "unit": "iter/sec",
            "range": "stddev: 0.000019939609988222094",
            "extra": "mean: 462.72433730032776 usec\nrounds: 2016"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 587.0605406018615,
            "unit": "iter/sec",
            "range": "stddev: 0.000034312717142589805",
            "extra": "mean: 1.7034018313933823 msec\nrounds: 516"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 604.5770480284431,
            "unit": "iter/sec",
            "range": "stddev: 0.00002964572101204788",
            "extra": "mean: 1.6540488979213677 msec\nrounds: 529"
          }
        ]
      }
    ],
    "GPU Benchmark Results": [
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6f2907205f7638575690e2a376cdee33821c28b1",
          "message": "[BugFix] Fix functorch dim mock (#777)",
          "timestamp": "2024-05-14T17:03:46+01:00",
          "tree_id": "57f2db169a1f1ffaae324b33349a757a62fb1626",
          "url": "https://github.com/pytorch/tensordict/commit/6f2907205f7638575690e2a376cdee33821c28b1"
        },
        "date": 1715703025067,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 72092.47679983781,
            "unit": "iter/sec",
            "range": "stddev: 7.725621174402465e-7",
            "extra": "mean: 13.871072882909326 usec\nrounds: 20182"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 72493.69731100528,
            "unit": "iter/sec",
            "range": "stddev: 6.541330425222192e-7",
            "extra": "mean: 13.7943026372334 usec\nrounds: 20450"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 66605.4695962627,
            "unit": "iter/sec",
            "range": "stddev: 7.31588081205575e-7",
            "extra": "mean: 15.013781992104008 usec\nrounds: 38196"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 66579.47204899635,
            "unit": "iter/sec",
            "range": "stddev: 6.799774736330382e-7",
            "extra": "mean: 15.019644482372769 usec\nrounds: 42195"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 212935.6182785801,
            "unit": "iter/sec",
            "range": "stddev: 3.1703867264618104e-7",
            "extra": "mean: 4.696255178369063 usec\nrounds: 91567"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 2972.9015264814557,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036616759599896824",
            "extra": "mean: 336.3717200493818 usec\nrounds: 2604"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 2970.652782547155,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036366870962664046",
            "extra": "mean: 336.6263488870485 usec\nrounds: 2832"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 12183.535379926738,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016018417573704276",
            "extra": "mean: 82.07798219616721 usec\nrounds: 9386"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 2978.943420071589,
            "unit": "iter/sec",
            "range": "stddev: 0.000003858819687437022",
            "extra": "mean: 335.68949086517677 usec\nrounds: 2740"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 11986.30369341911,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016025257330870866",
            "extra": "mean: 83.42855525586542 usec\nrounds: 9943"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 2942.2331632124537,
            "unit": "iter/sec",
            "range": "stddev: 0.000003763374302059801",
            "extra": "mean: 339.8778902036975 usec\nrounds: 2687"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 212363.35786460352,
            "unit": "iter/sec",
            "range": "stddev: 3.1418641510833864e-7",
            "extra": "mean: 4.708910284972843 usec\nrounds: 125945"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 14635.98019691093,
            "unit": "iter/sec",
            "range": "stddev: 0.000001643867250213407",
            "extra": "mean: 68.32477132013749 usec\nrounds: 11098"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 13440.95058001314,
            "unit": "iter/sec",
            "range": "stddev: 0.000018273204263047502",
            "extra": "mean: 74.39949980078138 usec\nrounds: 12037"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 16994.15982808727,
            "unit": "iter/sec",
            "range": "stddev: 0.000001447623805058371",
            "extra": "mean: 58.84374456377889 usec\nrounds: 14039"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 14676.922524348864,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015493995202087433",
            "extra": "mean: 68.13417447295305 usec\nrounds: 12535"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 17052.476180492788,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014623735825733103",
            "extra": "mean: 58.64250971034646 usec\nrounds: 14412"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 13724.198640439687,
            "unit": "iter/sec",
            "range": "stddev: 0.000001889889497297098",
            "extra": "mean: 72.86399929052344 usec\nrounds: 12660"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 551401.8580827411,
            "unit": "iter/sec",
            "range": "stddev: 1.07525138919585e-7",
            "extra": "mean: 1.8135593584632863 usec\nrounds: 175132"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 28005.057286124364,
            "unit": "iter/sec",
            "range": "stddev: 0.000001041597071920898",
            "extra": "mean: 35.7078362591127 usec\nrounds: 18536"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 26646.242395125413,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010430875866522745",
            "extra": "mean: 37.52874364690674 usec\nrounds: 23720"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 31331.21014853466,
            "unit": "iter/sec",
            "range": "stddev: 9.177147638648772e-7",
            "extra": "mean: 31.917056355602313 usec\nrounds: 24649"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 27935.55564801932,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010614976461100572",
            "extra": "mean: 35.79667476816062 usec\nrounds: 21213"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 31264.76978299639,
            "unit": "iter/sec",
            "range": "stddev: 9.081255204625943e-7",
            "extra": "mean: 31.984882887058983 usec\nrounds: 24190"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 26581.589614176843,
            "unit": "iter/sec",
            "range": "stddev: 0.00000105011152179636",
            "extra": "mean: 37.620022523659266 usec\nrounds: 24510"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 1431964.2797202438,
            "unit": "iter/sec",
            "range": "stddev: 2.663080120358571e-8",
            "extra": "mean: 698.3414420053588 nsec\nrounds: 69397"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 400550.8277370092,
            "unit": "iter/sec",
            "range": "stddev: 2.3957980870819305e-7",
            "extra": "mean: 2.496562060924195 usec\nrounds: 145561"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 399026.62892608123,
            "unit": "iter/sec",
            "range": "stddev: 2.53270888076223e-7",
            "extra": "mean: 2.5060984092498937 usec\nrounds: 187618"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 397896.58639765263,
            "unit": "iter/sec",
            "range": "stddev: 2.6814739333016973e-7",
            "extra": "mean: 2.513215830910932 usec\nrounds: 69253"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 400459.77993074054,
            "unit": "iter/sec",
            "range": "stddev: 2.440723075220806e-7",
            "extra": "mean: 2.4971296747277587 usec\nrounds: 90172"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 330876.16245682974,
            "unit": "iter/sec",
            "range": "stddev: 2.6801129024560424e-7",
            "extra": "mean: 3.0222787660941655 usec\nrounds: 145349"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 330140.49141104653,
            "unit": "iter/sec",
            "range": "stddev: 2.93988774851269e-7",
            "extra": "mean: 3.0290134837018052 usec\nrounds: 151976"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 330372.1373486021,
            "unit": "iter/sec",
            "range": "stddev: 3.269877465035491e-7",
            "extra": "mean: 3.0268896403476657 usec\nrounds: 62696"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 329864.6317384812,
            "unit": "iter/sec",
            "range": "stddev: 2.9662309324029654e-7",
            "extra": "mean: 3.031546591490313 usec\nrounds: 96994"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 119934.41799608406,
            "unit": "iter/sec",
            "range": "stddev: 5.214900646740552e-7",
            "extra": "mean: 8.337890129526045 usec\nrounds: 84675"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 127264.78978237793,
            "unit": "iter/sec",
            "range": "stddev: 4.6259708600678944e-7",
            "extra": "mean: 7.857632906242129 usec\nrounds: 90992"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 119887.9685679104,
            "unit": "iter/sec",
            "range": "stddev: 4.1871472808409626e-7",
            "extra": "mean: 8.341120564016824 usec\nrounds: 49408"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 126951.79149370945,
            "unit": "iter/sec",
            "range": "stddev: 3.910698364551276e-7",
            "extra": "mean: 7.877005816413003 usec\nrounds: 65613"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 117299.64954954117,
            "unit": "iter/sec",
            "range": "stddev: 4.111789420468767e-7",
            "extra": "mean: 8.525174660284495 usec\nrounds: 65058"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 124662.51396026678,
            "unit": "iter/sec",
            "range": "stddev: 4.283511213713367e-7",
            "extra": "mean: 8.021657579589052 usec\nrounds: 83258"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 117372.21651566269,
            "unit": "iter/sec",
            "range": "stddev: 4.7048078247579995e-7",
            "extra": "mean: 8.519903855326405 usec\nrounds: 55739"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 124138.12405356998,
            "unit": "iter/sec",
            "range": "stddev: 4.762833499727726e-7",
            "extra": "mean: 8.05554303018519 usec\nrounds: 69109"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2426.114413980644,
            "unit": "iter/sec",
            "range": "stddev: 0.001747991185942694",
            "extra": "mean: 412.1817150244169 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3186.356541597395,
            "unit": "iter/sec",
            "range": "stddev: 0.000005635993024880254",
            "extra": "mean: 313.83807397105556 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2771.894516388141,
            "unit": "iter/sec",
            "range": "stddev: 0.00007619022717273952",
            "extra": "mean: 360.76408899680246 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3103.9393929220064,
            "unit": "iter/sec",
            "range": "stddev: 0.0000062465216137517305",
            "extra": "mean: 322.1712390004541 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 9869.9809735375,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028560436448020332",
            "extra": "mean: 101.3173179037639 usec\nrounds: 6694"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 3440.847803958913,
            "unit": "iter/sec",
            "range": "stddev: 0.000004197468458205254",
            "extra": "mean: 290.626048280728 usec\nrounds: 3065"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1630.5967585758697,
            "unit": "iter/sec",
            "range": "stddev: 0.00006391289411571321",
            "extra": "mean: 613.2724076266286 usec\nrounds: 1180"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 621315.2525618195,
            "unit": "iter/sec",
            "range": "stddev: 2.178879251979355e-7",
            "extra": "mean: 1.6094888961389247 usec\nrounds: 106724"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 97221.09167862513,
            "unit": "iter/sec",
            "range": "stddev: 5.893544433383186e-7",
            "extra": "mean: 10.285833894003252 usec\nrounds: 32949"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 83738.89809472662,
            "unit": "iter/sec",
            "range": "stddev: 6.457248762595446e-7",
            "extra": "mean: 11.941881524029439 usec\nrounds: 33146"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 70119.84432964644,
            "unit": "iter/sec",
            "range": "stddev: 7.504480733072564e-7",
            "extra": "mean: 14.261298061342147 usec\nrounds: 33863"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 80890.21227333896,
            "unit": "iter/sec",
            "range": "stddev: 8.726239320835274e-7",
            "extra": "mean: 12.362435106745238 usec\nrounds: 678"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 88331.3722978877,
            "unit": "iter/sec",
            "range": "stddev: 7.86033204903222e-7",
            "extra": "mean: 11.32100604785819 usec\nrounds: 14897"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 46221.402570370745,
            "unit": "iter/sec",
            "range": "stddev: 0.000001229495420800469",
            "extra": "mean: 21.63499903486332 usec\nrounds: 11295"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 20756.878746978156,
            "unit": "iter/sec",
            "range": "stddev: 0.000010719927559392733",
            "extra": "mean: 48.1768001918681 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 50121.449858963824,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012063627691896627",
            "extra": "mean: 19.951537771031933 usec\nrounds: 17461"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 27791.531315534547,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030747441829156985",
            "extra": "mean: 35.98218423613934 usec\nrounds: 12479"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 31970.599350578195,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015645900331471186",
            "extra": "mean: 31.27873797529901 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 19594.08618294182,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022768663934939105",
            "extra": "mean: 51.03580696049903 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 14214.494172952594,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027839728702571397",
            "extra": "mean: 70.35072707003565 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 21750.5755002445,
            "unit": "iter/sec",
            "range": "stddev: 0.000002396423009509295",
            "extra": "mean: 45.97579498476989 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 55544.87660815813,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015510726492611117",
            "extra": "mean: 18.003460644165436 usec\nrounds: 16548"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 58083.84007151682,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015197923445263833",
            "extra": "mean: 17.216492552295634 usec\nrounds: 24090"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 10066.640848691084,
            "unit": "iter/sec",
            "range": "stddev: 0.00006294539536560164",
            "extra": "mean: 99.33800311650386 usec\nrounds: 4816"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 49559.556965421296,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017641905839092016",
            "extra": "mean: 20.17774292651002 usec\nrounds: 29673"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 38826.454896953765,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017037583777046306",
            "extra": "mean: 25.7556349827462 usec\nrounds: 18325"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 43190.04150430849,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016129643786422727",
            "extra": "mean: 23.153485506612526 usec\nrounds: 13530"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 54792.19516727313,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015348058179805899",
            "extra": "mean: 18.25077452996975 usec\nrounds: 21515"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 47021.004936772326,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016932581617401413",
            "extra": "mean: 21.26709119349254 usec\nrounds: 15440"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 29085.755350138894,
            "unit": "iter/sec",
            "range": "stddev: 0.000008631906869482487",
            "extra": "mean: 34.38109094853625 usec\nrounds: 12755"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 18274.256134539824,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017877123465827815",
            "extra": "mean: 54.72178963880882 usec\nrounds: 13733"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 9062.2376803133,
            "unit": "iter/sec",
            "range": "stddev: 0.000004019110747149247",
            "extra": "mean: 110.34802167817651 usec\nrounds: 6325"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2899.1853497363645,
            "unit": "iter/sec",
            "range": "stddev: 0.000005329541463383653",
            "extra": "mean: 344.92448028234355 usec\nrounds: 2467"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 1144138.9813143439,
            "unit": "iter/sec",
            "range": "stddev: 4.567570668452687e-8",
            "extra": "mean: 874.0196919531905 nsec\nrounds: 108814"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to",
            "value": 12785.659548639478,
            "unit": "iter/sec",
            "range": "stddev: 0.000010734341521628517",
            "extra": "mean: 78.21262533980189 usec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_nonblocking",
            "value": 15807.883373012215,
            "unit": "iter/sec",
            "range": "stddev: 0.000002785019356705431",
            "extra": "mean: 63.2595760231402 usec\nrounds: 9213"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3656.5554839441033,
            "unit": "iter/sec",
            "range": "stddev: 0.00003757753123656582",
            "extra": "mean: 273.48142381292706 usec\nrounds: 1069"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 3628.938920944572,
            "unit": "iter/sec",
            "range": "stddev: 0.000004749127341775319",
            "extra": "mean: 275.56264290601814 usec\nrounds: 3366"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1232.5298532258844,
            "unit": "iter/sec",
            "range": "stddev: 0.002711350319715401",
            "extra": "mean: 811.3393743629925 usec\nrounds: 1365"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 639.5418636736512,
            "unit": "iter/sec",
            "range": "stddev: 0.000010800125485486534",
            "extra": "mean: 1.5636192981266437 msec\nrounds: 597"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 593.3255645341494,
            "unit": "iter/sec",
            "range": "stddev: 0.0028705787256793863",
            "extra": "mean: 1.6854153263818183 msec\nrounds: 619"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 13099.323126860032,
            "unit": "iter/sec",
            "range": "stddev: 0.000005797287292820999",
            "extra": "mean: 76.33982231872042 usec\nrounds: 3022"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 18957.760121542953,
            "unit": "iter/sec",
            "range": "stddev: 0.000004563020829359581",
            "extra": "mean: 52.74884762697435 usec\nrounds: 10962"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 141728.81266472593,
            "unit": "iter/sec",
            "range": "stddev: 0.00000144971404361995",
            "extra": "mean: 7.055728339202295 usec\nrounds: 25044"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1475831.5718798821,
            "unit": "iter/sec",
            "range": "stddev: 1.1214636110912769e-7",
            "extra": "mean: 677.5840949968441 nsec\nrounds: 89687"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 200710.65915668054,
            "unit": "iter/sec",
            "range": "stddev: 5.471679192250454e-7",
            "extra": "mean: 4.982296427113874 usec\nrounds: 28522"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 3463.6477116413594,
            "unit": "iter/sec",
            "range": "stddev: 0.00006521310782479944",
            "extra": "mean: 288.7129648431013 usec\nrounds: 796"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 2779.9384551835424,
            "unit": "iter/sec",
            "range": "stddev: 0.00006356801435852374",
            "extra": "mean: 359.7201938537075 usec\nrounds: 1357"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1443.3503741462712,
            "unit": "iter/sec",
            "range": "stddev: 0.00007089783491780057",
            "extra": "mean: 692.8324666777401 usec\nrounds: 1395"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 8.87892383586561,
            "unit": "iter/sec",
            "range": "stddev: 0.029183595202035725",
            "extra": "mean: 112.62626175039259 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 0.806372403397431,
            "unit": "iter/sec",
            "range": "stddev: 0.23943632824401856",
            "extra": "mean: 1.2401218044997222 sec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.774891211894959,
            "unit": "iter/sec",
            "range": "stddev: 0.0015903716436256824",
            "extra": "mean: 102.30292883291743 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 11.530143692749647,
            "unit": "iter/sec",
            "range": "stddev: 0.03370534462307882",
            "extra": "mean: 86.72918800039042 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 0.8008695938863564,
            "unit": "iter/sec",
            "range": "stddev: 0.2313530021005633",
            "extra": "mean: 1.2486427348893712 sec\nrounds: 9"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 41602.401698539754,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012245629201211459",
            "extra": "mean: 24.03707380276317 usec\nrounds: 10525"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 32064.051892488522,
            "unit": "iter/sec",
            "range": "stddev: 0.000001411846546057745",
            "extra": "mean: 31.187574276420904 usec\nrounds: 8865"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 42095.05548269299,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011349130296962768",
            "extra": "mean: 23.755759163000537 usec\nrounds: 21240"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 28465.99249815158,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024730793530068026",
            "extra": "mean: 35.12963758649674 usec\nrounds: 13198"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 33510.16563539596,
            "unit": "iter/sec",
            "range": "stddev: 0.000001340012619157813",
            "extra": "mean: 29.84169075379695 usec\nrounds: 14765"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 23626.80984505501,
            "unit": "iter/sec",
            "range": "stddev: 0.000004416152320262824",
            "extra": "mean: 42.32479994370868 usec\nrounds: 13936"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 30998.434027835978,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015397773782718708",
            "extra": "mean: 32.259694122032734 usec\nrounds: 9612"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 25544.59416102132,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019364094299399736",
            "extra": "mean: 39.147225972605476 usec\nrounds: 12364"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 27585.137821055017,
            "unit": "iter/sec",
            "range": "stddev: 0.00000259702221453851",
            "extra": "mean: 36.25140488646484 usec\nrounds: 10981"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 17725.981975366623,
            "unit": "iter/sec",
            "range": "stddev: 0.000002973137296575406",
            "extra": "mean: 56.41436403295887 usec\nrounds: 9933"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 14756.547871865469,
            "unit": "iter/sec",
            "range": "stddev: 0.00000864252217120107",
            "extra": "mean: 67.7665270145316 usec\nrounds: 4427"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 63481.82570381928,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023377769283475194",
            "extra": "mean: 15.752540020912422 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 31997.41556135844,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016669980912912187",
            "extra": "mean: 31.25252406971413 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 56788.30992202762,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012202339741748852",
            "extra": "mean: 17.609257985895965 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 28513.250966628428,
            "unit": "iter/sec",
            "range": "stddev: 0.000002814575530475105",
            "extra": "mean: 35.07141297814087 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 654.7446713384485,
            "unit": "iter/sec",
            "range": "stddev: 0.000020007798094239964",
            "extra": "mean: 1.527312926359172 msec\nrounds: 529"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 937.424572569939,
            "unit": "iter/sec",
            "range": "stddev: 0.00009657468124217056",
            "extra": "mean: 1.0667524932256804 msec\nrounds: 890"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6639.465026870769,
            "unit": "iter/sec",
            "range": "stddev: 0.000004229220324652575",
            "extra": "mean: 150.6145443876685 usec\nrounds: 2770"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 7197.579656096977,
            "unit": "iter/sec",
            "range": "stddev: 0.000004769210368607526",
            "extra": "mean: 138.93559332169573 usec\nrounds: 4475"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 7194.113201499642,
            "unit": "iter/sec",
            "range": "stddev: 0.000004564009359327155",
            "extra": "mean: 139.00253888019805 usec\nrounds: 2713"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4769.026170719434,
            "unit": "iter/sec",
            "range": "stddev: 0.000013597407717030673",
            "extra": "mean: 209.68641483658382 usec\nrounds: 3112"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 1662.4414226897165,
            "unit": "iter/sec",
            "range": "stddev: 0.00001248183486055653",
            "extra": "mean: 601.5249538128497 usec\nrounds: 1256"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 1651.9785559580666,
            "unit": "iter/sec",
            "range": "stddev: 0.000014591020262510655",
            "extra": "mean: 605.3347341546144 usec\nrounds: 1625"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 1881.1104179944125,
            "unit": "iter/sec",
            "range": "stddev: 0.000013197271490198328",
            "extra": "mean: 531.6009046753204 usec\nrounds: 1794"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 1894.937983262624,
            "unit": "iter/sec",
            "range": "stddev: 0.00001146225086332323",
            "extra": "mean: 527.7217559797089 usec\nrounds: 1873"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1498.7187191182918,
            "unit": "iter/sec",
            "range": "stddev: 0.000021300684731781234",
            "extra": "mean: 667.2366116760775 usec\nrounds: 1146"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1502.744636451256,
            "unit": "iter/sec",
            "range": "stddev: 0.000014768949097496752",
            "extra": "mean: 665.4490561759771 usec\nrounds: 1478"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1702.932105920325,
            "unit": "iter/sec",
            "range": "stddev: 0.00001369857349743824",
            "extra": "mean: 587.2224714793105 usec\nrounds: 1663"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1689.6700955339852,
            "unit": "iter/sec",
            "range": "stddev: 0.00001679842970827427",
            "extra": "mean: 591.8315076079812 usec\nrounds: 1698"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed[True-True]",
            "value": 126.24200393199312,
            "unit": "iter/sec",
            "range": "stddev: 0.00005766978746395357",
            "extra": "mean: 7.921293775871163 msec\nrounds: 125"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed[True-False]",
            "value": 126.43398035710844,
            "unit": "iter/sec",
            "range": "stddev: 0.0000487161400191607",
            "extra": "mean: 7.9092661417091685 msec\nrounds: 127"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed[False-True]",
            "value": 127.63427050886452,
            "unit": "iter/sec",
            "range": "stddev: 0.000049358169112875875",
            "extra": "mean: 7.834886320210899 msec\nrounds: 128"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed[False-False]",
            "value": 127.7298437139018,
            "unit": "iter/sec",
            "range": "stddev: 0.000026493050904822484",
            "extra": "mean: 7.8290239064244815 msec\nrounds: 128"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-True]",
            "value": 52.100539714383686,
            "unit": "iter/sec",
            "range": "stddev: 0.00010722824449781125",
            "extra": "mean: 19.19365913447389 msec\nrounds: 52"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-False]",
            "value": 52.0448770483851,
            "unit": "iter/sec",
            "range": "stddev: 0.00005214347139880147",
            "extra": "mean: 19.21418700000616 msec\nrounds: 52"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-True]",
            "value": 52.200833067711656,
            "unit": "iter/sec",
            "range": "stddev: 0.00014449045409071422",
            "extra": "mean: 19.15678239661161 msec\nrounds: 53"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-False]",
            "value": 52.395052695093845,
            "unit": "iter/sec",
            "range": "stddev: 0.00004490883960364377",
            "extra": "mean: 19.08577143379107 msec\nrounds: 53"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 646.8161934968555,
            "unit": "iter/sec",
            "range": "stddev: 0.000032428284412795224",
            "extra": "mean: 1.5460342676236065 msec\nrounds: 624"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 660.9680123513943,
            "unit": "iter/sec",
            "range": "stddev: 0.000021539601821012738",
            "extra": "mean: 1.5129325191433984 msec\nrounds: 601"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5fef53825ae0034b442ba6e3bfbaa6933625fa71",
          "message": "[Feature] online edition of memory mapped tensordicts (#775)",
          "timestamp": "2024-05-14T18:03:22+01:00",
          "tree_id": "26e5300a78f6d4a498d8f887f72214047cf6dd02",
          "url": "https://github.com/pytorch/tensordict/commit/5fef53825ae0034b442ba6e3bfbaa6933625fa71"
        },
        "date": 1715706592688,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 76864.46052183831,
            "unit": "iter/sec",
            "range": "stddev: 8.019860022028251e-7",
            "extra": "mean: 13.009913726199711 usec\nrounds: 20121"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 76849.04217798612,
            "unit": "iter/sec",
            "range": "stddev: 7.386132881965092e-7",
            "extra": "mean: 13.01252392559365 usec\nrounds: 20471"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 69834.42012208604,
            "unit": "iter/sec",
            "range": "stddev: 7.721418731224556e-7",
            "extra": "mean: 14.319586219113416 usec\nrounds: 38198"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 69466.06809255773,
            "unit": "iter/sec",
            "range": "stddev: 7.159497313690108e-7",
            "extra": "mean: 14.39551751608546 usec\nrounds: 43009"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 209664.1060249498,
            "unit": "iter/sec",
            "range": "stddev: 3.1372865876463193e-7",
            "extra": "mean: 4.769533607631442 usec\nrounds: 95603"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 2947.5807312602383,
            "unit": "iter/sec",
            "range": "stddev: 0.0000041193747923650694",
            "extra": "mean: 339.26127599987734 usec\nrounds: 2652"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 2923.235267211853,
            "unit": "iter/sec",
            "range": "stddev: 0.000004184868244771514",
            "extra": "mean: 342.0867321957935 usec\nrounds: 2685"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 12147.462090759716,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018085842871057057",
            "extra": "mean: 82.32172222712067 usec\nrounds: 9635"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 2982.7938369471617,
            "unit": "iter/sec",
            "range": "stddev: 0.000004245517353623809",
            "extra": "mean: 335.2561573693886 usec\nrounds: 2764"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 12139.161148615218,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019155389114559967",
            "extra": "mean: 82.37801506688751 usec\nrounds: 10345"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 2908.67417246541,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035844998991243545",
            "extra": "mean: 343.79925034793223 usec\nrounds: 2792"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 212122.59815018772,
            "unit": "iter/sec",
            "range": "stddev: 3.234174255223115e-7",
            "extra": "mean: 4.714254910700165 usec\nrounds: 124549"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 14313.2157579571,
            "unit": "iter/sec",
            "range": "stddev: 0.000001725753290628854",
            "extra": "mean: 69.86550170908122 usec\nrounds: 10798"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 13309.035061306171,
            "unit": "iter/sec",
            "range": "stddev: 0.000018565797300628872",
            "extra": "mean: 75.13692731243421 usec\nrounds: 11939"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 16631.437450089823,
            "unit": "iter/sec",
            "range": "stddev: 0.00000162419546960558",
            "extra": "mean: 60.12709382462904 usec\nrounds: 13953"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 14577.493825513793,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017837624183023353",
            "extra": "mean: 68.5988971746146 usec\nrounds: 12612"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 16608.174938288706,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016444145309045995",
            "extra": "mean: 60.211311821781614 usec\nrounds: 13973"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 13521.736147563686,
            "unit": "iter/sec",
            "range": "stddev: 0.000002077763451025333",
            "extra": "mean: 73.95500023716833 usec\nrounds: 12584"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 544620.8174071969,
            "unit": "iter/sec",
            "range": "stddev: 1.3424550706376526e-7",
            "extra": "mean: 1.8361398757409777 usec\nrounds: 197590"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 28032.52178522017,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011400595786674137",
            "extra": "mean: 35.67285197035818 usec\nrounds: 19081"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 26392.112514287608,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013257497196015146",
            "extra": "mean: 37.89010824573595 usec\nrounds: 23685"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 31337.047663948004,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011396768341095636",
            "extra": "mean: 31.91111079524123 usec\nrounds: 25063"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 27430.70783893049,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013154204660078048",
            "extra": "mean: 36.45549381634147 usec\nrounds: 20925"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 30970.807965803884,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011537292691858429",
            "extra": "mean: 32.28846987473302 usec\nrounds: 24558"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 25901.897874505765,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013187743611863375",
            "extra": "mean: 38.60720958923482 usec\nrounds: 23832"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 1409294.5112449143,
            "unit": "iter/sec",
            "range": "stddev: 2.981576093568076e-8",
            "extra": "mean: 709.5748915651704 nsec\nrounds: 66841"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 393010.7647713943,
            "unit": "iter/sec",
            "range": "stddev: 2.7670137833728937e-7",
            "extra": "mean: 2.5444595660927454 usec\nrounds: 148589"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 389706.99489789933,
            "unit": "iter/sec",
            "range": "stddev: 2.6587043758551077e-7",
            "extra": "mean: 2.566030410262442 usec\nrounds: 185840"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 396480.0908703761,
            "unit": "iter/sec",
            "range": "stddev: 3.092105038500909e-7",
            "extra": "mean: 2.5221947407365195 usec\nrounds: 87874"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 391339.06593138544,
            "unit": "iter/sec",
            "range": "stddev: 2.754404703794226e-7",
            "extra": "mean: 2.5553288364400935 usec\nrounds: 114156"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 315547.2895335256,
            "unit": "iter/sec",
            "range": "stddev: 3.1033704383436025e-7",
            "extra": "mean: 3.169097099449983 usec\nrounds: 144280"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 318556.6178058859,
            "unit": "iter/sec",
            "range": "stddev: 3.1010019400430946e-7",
            "extra": "mean: 3.1391593961779036 usec\nrounds: 153351"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 99974.38767385723,
            "unit": "iter/sec",
            "range": "stddev: 6.12594932849986e-7",
            "extra": "mean: 10.002561888773586 usec\nrounds: 41686"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 100152.22176687953,
            "unit": "iter/sec",
            "range": "stddev: 5.985676874246504e-7",
            "extra": "mean: 9.984800959559955 usec\nrounds: 51895"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 118216.70017469685,
            "unit": "iter/sec",
            "range": "stddev: 4.762956566000958e-7",
            "extra": "mean: 8.459041730332789 usec\nrounds: 85187"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 125788.45281358936,
            "unit": "iter/sec",
            "range": "stddev: 4.531363575511583e-7",
            "extra": "mean: 7.949855313682391 usec\nrounds: 92166"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 118779.87467042352,
            "unit": "iter/sec",
            "range": "stddev: 5.06482956543244e-7",
            "extra": "mean: 8.418934628232963 usec\nrounds: 59989"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 126756.01356714539,
            "unit": "iter/sec",
            "range": "stddev: 4.373569102700696e-7",
            "extra": "mean: 7.889172054707119 usec\nrounds: 68682"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 116466.87862562823,
            "unit": "iter/sec",
            "range": "stddev: 4.795663334169918e-7",
            "extra": "mean: 8.586132055744411 usec\nrounds: 62228"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 124262.85759035508,
            "unit": "iter/sec",
            "range": "stddev: 4.7249249392619443e-7",
            "extra": "mean: 8.0474569746062 usec\nrounds: 84883"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 116549.92194047412,
            "unit": "iter/sec",
            "range": "stddev: 4.941029470746157e-7",
            "extra": "mean: 8.580014326485204 usec\nrounds: 57274"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 124485.84093867685,
            "unit": "iter/sec",
            "range": "stddev: 4.608867186402493e-7",
            "extra": "mean: 8.033042091048824 usec\nrounds: 68353"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2389.353456105479,
            "unit": "iter/sec",
            "range": "stddev: 0.0017635859608996244",
            "extra": "mean: 418.5232609452214 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3255.799770735087,
            "unit": "iter/sec",
            "range": "stddev: 0.000006345466297836386",
            "extra": "mean: 307.14419510331936 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2776.100891139726,
            "unit": "iter/sec",
            "range": "stddev: 0.00007756560140205215",
            "extra": "mean: 360.2174557818216 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3192.243869984071,
            "unit": "iter/sec",
            "range": "stddev: 0.0000065643726209378675",
            "extra": "mean: 313.25927489524474 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 9731.457481185005,
            "unit": "iter/sec",
            "range": "stddev: 0.000003419055827843162",
            "extra": "mean: 102.759530310174 usec\nrounds: 6837"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 3392.237754997349,
            "unit": "iter/sec",
            "range": "stddev: 0.000004993579950488741",
            "extra": "mean: 294.79065803298374 usec\nrounds: 3161"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1720.7064463650966,
            "unit": "iter/sec",
            "range": "stddev: 0.00006329434592587945",
            "extra": "mean: 581.1566534852288 usec\nrounds: 1235"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 605348.7458279004,
            "unit": "iter/sec",
            "range": "stddev: 2.1340064658124068e-7",
            "extra": "mean: 1.6519403185222725 usec\nrounds: 137912"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 113538.72071978319,
            "unit": "iter/sec",
            "range": "stddev: 6.390005792780864e-7",
            "extra": "mean: 8.807567970296484 usec\nrounds: 37244"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 94371.46327529577,
            "unit": "iter/sec",
            "range": "stddev: 7.126154011710968e-7",
            "extra": "mean: 10.596423593463307 usec\nrounds: 36752"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 76656.87701944409,
            "unit": "iter/sec",
            "range": "stddev: 8.04729085828377e-7",
            "extra": "mean: 13.04514400901499 usec\nrounds: 34818"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 83724.2456727089,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015747943577215796",
            "extra": "mean: 11.943971450147853 usec\nrounds: 18540"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 87068.40268305915,
            "unit": "iter/sec",
            "range": "stddev: 6.715624409565601e-7",
            "extra": "mean: 11.485222758020912 usec\nrounds: 14687"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 45264.54784094933,
            "unit": "iter/sec",
            "range": "stddev: 0.000001192287626521319",
            "extra": "mean: 22.092344841570103 usec\nrounds: 11850"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 20603.009200339016,
            "unit": "iter/sec",
            "range": "stddev: 0.000011620240402670034",
            "extra": "mean: 48.53659920627251 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 49384.28807794641,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011399381407713412",
            "extra": "mean: 20.249355390557323 usec\nrounds: 16990"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 28280.523391420083,
            "unit": "iter/sec",
            "range": "stddev: 0.00000341013939054997",
            "extra": "mean: 35.36002450023206 usec\nrounds: 13209"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 33197.306159732325,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011392944621511038",
            "extra": "mean: 30.122926094918512 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 19904.776329986405,
            "unit": "iter/sec",
            "range": "stddev: 0.000001758594459178538",
            "extra": "mean: 50.23919804079924 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 14597.585390405144,
            "unit": "iter/sec",
            "range": "stddev: 0.000003128773268293378",
            "extra": "mean: 68.50448024488287 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 22889.58011883547,
            "unit": "iter/sec",
            "range": "stddev: 0.000001953014008175173",
            "extra": "mean: 43.68800103839021 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 58455.845838617424,
            "unit": "iter/sec",
            "range": "stddev: 0.000001472113379232103",
            "extra": "mean: 17.10692892479497 usec\nrounds: 22262"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 61028.69273184409,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013566407760869136",
            "extra": "mean: 16.38573522120049 usec\nrounds: 24289"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 10124.262632005162,
            "unit": "iter/sec",
            "range": "stddev: 0.00006481731062507357",
            "extra": "mean: 98.77262536026733 usec\nrounds: 4912"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 53703.24634316456,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016348935643678467",
            "extra": "mean: 18.620848237180763 usec\nrounds: 32259"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 42597.08105900665,
            "unit": "iter/sec",
            "range": "stddev: 0.000001787569701735779",
            "extra": "mean: 23.47578695861279 usec\nrounds: 23901"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 44513.015000314655,
            "unit": "iter/sec",
            "range": "stddev: 0.000001632591749521324",
            "extra": "mean: 22.465339631407378 usec\nrounds: 14577"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 57467.00315855548,
            "unit": "iter/sec",
            "range": "stddev: 0.000001499104338958711",
            "extra": "mean: 17.401290219379113 usec\nrounds: 24432"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 50196.980589237384,
            "unit": "iter/sec",
            "range": "stddev: 0.000001512156484179599",
            "extra": "mean: 19.92151695702605 usec\nrounds: 20760"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 30642.362839200872,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019630626742937513",
            "extra": "mean: 32.634559066074914 usec\nrounds: 13901"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 17973.622524242484,
            "unit": "iter/sec",
            "range": "stddev: 0.000008565523373602581",
            "extra": "mean: 55.6370869951908 usec\nrounds: 13641"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 8881.65475267076,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034937754428555956",
            "extra": "mean: 112.59163161000991 usec\nrounds: 6269"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2848.783611574426,
            "unit": "iter/sec",
            "range": "stddev: 0.0000056581616250445865",
            "extra": "mean: 351.02701234908255 usec\nrounds: 2582"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 1143666.2354372067,
            "unit": "iter/sec",
            "range": "stddev: 4.685034428937016e-8",
            "extra": "mean: 874.3809767346282 nsec\nrounds: 108933"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to",
            "value": 12774.267747400501,
            "unit": "iter/sec",
            "range": "stddev: 0.000010898050140722315",
            "extra": "mean: 78.28237357898615 usec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_nonblocking",
            "value": 16130.463778495765,
            "unit": "iter/sec",
            "range": "stddev: 0.000002936074274038133",
            "extra": "mean: 61.99449772381277 usec\nrounds: 9131"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3633.3584135569304,
            "unit": "iter/sec",
            "range": "stddev: 0.0000075896408656123155",
            "extra": "mean: 275.22745795426084 usec\nrounds: 533"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 3694.5056710380313,
            "unit": "iter/sec",
            "range": "stddev: 0.000006555490488744277",
            "extra": "mean: 270.672205983929 usec\nrounds: 3436"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1258.6085903836781,
            "unit": "iter/sec",
            "range": "stddev: 0.0026730021111315847",
            "extra": "mean: 794.5281858398542 usec\nrounds: 1405"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 600.7674830955866,
            "unit": "iter/sec",
            "range": "stddev: 0.0028611184867626746",
            "extra": "mean: 1.6645374926873209 msec\nrounds: 609"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 649.955528562087,
            "unit": "iter/sec",
            "range": "stddev: 0.000015301444374129305",
            "extra": "mean: 1.5385668035047337 msec\nrounds: 621"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 17429.76229800596,
            "unit": "iter/sec",
            "range": "stddev: 0.000004692464666596509",
            "extra": "mean: 57.37312895623392 usec\nrounds: 3279"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 18801.770234251126,
            "unit": "iter/sec",
            "range": "stddev: 0.0000050210718326712105",
            "extra": "mean: 53.18648124836156 usec\nrounds: 7863"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 143518.39373123075,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016706501425361872",
            "extra": "mean: 6.967747993840541 usec\nrounds: 30248"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1454213.1204963694,
            "unit": "iter/sec",
            "range": "stddev: 1.2601003000634513e-7",
            "extra": "mean: 687.657115663121 nsec\nrounds: 123763"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 216515.7487320308,
            "unit": "iter/sec",
            "range": "stddev: 6.094691222863482e-7",
            "extra": "mean: 4.61860167611938 usec\nrounds: 28280"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 3512.7551906872072,
            "unit": "iter/sec",
            "range": "stddev: 0.00007246855257466486",
            "extra": "mean: 284.67682651245843 usec\nrounds: 669"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 2776.5741298630455,
            "unit": "iter/sec",
            "range": "stddev: 0.00006357203528241842",
            "extra": "mean: 360.1560603927852 usec\nrounds: 2446"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1493.7423650192568,
            "unit": "iter/sec",
            "range": "stddev: 0.00006955140532948441",
            "extra": "mean: 669.4594887433003 usec\nrounds: 1473"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 9.802349220680524,
            "unit": "iter/sec",
            "range": "stddev: 0.002121499227194335",
            "extra": "mean: 102.01636133206193 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 0.8083601118494597,
            "unit": "iter/sec",
            "range": "stddev: 0.25114186050097903",
            "extra": "mean: 1.2370724202510246 sec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.220192336827733,
            "unit": "iter/sec",
            "range": "stddev: 0.027306374511207314",
            "extra": "mean: 108.45760733273993 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 11.14573294377619,
            "unit": "iter/sec",
            "range": "stddev: 0.03554390361303059",
            "extra": "mean: 89.7204342724184 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 0.8075916004956479,
            "unit": "iter/sec",
            "range": "stddev: 0.25663558099148603",
            "extra": "mean: 1.2382496293748773 sec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 42580.22532591772,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025021529419954346",
            "extra": "mean: 23.48508004233881 usec\nrounds: 10635"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 31689.348543043852,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016616684675835697",
            "extra": "mean: 31.55634451246902 usec\nrounds: 8792"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 43508.86719833159,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013665500473648913",
            "extra": "mean: 22.983820641470217 usec\nrounds: 20211"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 26946.52172066631,
            "unit": "iter/sec",
            "range": "stddev: 0.0000046781650611585355",
            "extra": "mean: 37.110541032576464 usec\nrounds: 12329"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 33364.951705331536,
            "unit": "iter/sec",
            "range": "stddev: 0.000001513331583866301",
            "extra": "mean: 29.971570432101224 usec\nrounds: 12977"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 23320.76954289962,
            "unit": "iter/sec",
            "range": "stddev: 0.000004663170499978012",
            "extra": "mean: 42.88023163903122 usec\nrounds: 13606"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 31325.555625167606,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016269275592712781",
            "extra": "mean: 31.92281764977152 usec\nrounds: 9174"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 24734.271274727085,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034433275287692898",
            "extra": "mean: 40.42973366358188 usec\nrounds: 11336"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 27516.849072625926,
            "unit": "iter/sec",
            "range": "stddev: 0.000006416650872186108",
            "extra": "mean: 36.341370240490626 usec\nrounds: 10845"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 17762.248782013427,
            "unit": "iter/sec",
            "range": "stddev: 0.00000867823897701465",
            "extra": "mean: 56.29917767015117 usec\nrounds: 9632"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 11087.885011337126,
            "unit": "iter/sec",
            "range": "stddev: 0.00022556333517614538",
            "extra": "mean: 90.18852549223962 usec\nrounds: 4640"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 62554.215485286426,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025028382786141376",
            "extra": "mean: 15.98613286478212 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 32534.25514689947,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020311289830756546",
            "extra": "mean: 30.736834007257126 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 56576.83653253232,
            "unit": "iter/sec",
            "range": "stddev: 0.000001573065407480076",
            "extra": "mean: 17.675078022875823 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 28404.728477987574,
            "unit": "iter/sec",
            "range": "stddev: 0.000002179844559610475",
            "extra": "mean: 35.205406056775246 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 636.8559916687728,
            "unit": "iter/sec",
            "range": "stddev: 0.000015130419757957225",
            "extra": "mean: 1.570213695218082 msec\nrounds: 548"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 843.8315879488194,
            "unit": "iter/sec",
            "range": "stddev: 0.002549097421867136",
            "extra": "mean: 1.18507059261765 msec\nrounds: 854"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6455.5864974947135,
            "unit": "iter/sec",
            "range": "stddev: 0.000004554951966712961",
            "extra": "mean: 154.9045931594411 usec\nrounds: 2642"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 7032.226056423452,
            "unit": "iter/sec",
            "range": "stddev: 0.00000507385405775764",
            "extra": "mean: 142.20248211255512 usec\nrounds: 4314"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 6986.779649451323,
            "unit": "iter/sec",
            "range": "stddev: 0.000005506536527646928",
            "extra": "mean: 143.1274564496292 usec\nrounds: 2592"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4656.129210404774,
            "unit": "iter/sec",
            "range": "stddev: 0.000013626321319388038",
            "extra": "mean: 214.77067212081653 usec\nrounds: 2946"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 1619.4065901302,
            "unit": "iter/sec",
            "range": "stddev: 0.000010869390166436948",
            "extra": "mean: 617.5101460588722 usec\nrounds: 1212"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 1589.4109262138522,
            "unit": "iter/sec",
            "range": "stddev: 0.000018790942564386328",
            "extra": "mean: 629.1639144460315 usec\nrounds: 1575"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 1808.2225201837855,
            "unit": "iter/sec",
            "range": "stddev: 0.000016833522238785565",
            "extra": "mean: 553.0292808754318 usec\nrounds: 1798"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 1785.6373142988448,
            "unit": "iter/sec",
            "range": "stddev: 0.00001384504116155873",
            "extra": "mean: 560.0241392763814 usec\nrounds: 1745"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1468.567426672469,
            "unit": "iter/sec",
            "range": "stddev: 0.000021161809011968805",
            "extra": "mean: 680.9357077092706 usec\nrounds: 1177"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1461.464679139873,
            "unit": "iter/sec",
            "range": "stddev: 0.000016297253075851426",
            "extra": "mean: 684.2450688500646 usec\nrounds: 1452"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1647.7521472003104,
            "unit": "iter/sec",
            "range": "stddev: 0.00002360578201564926",
            "extra": "mean: 606.8873900113534 usec\nrounds: 1615"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1630.2880638302454,
            "unit": "iter/sec",
            "range": "stddev: 0.000025847413810610898",
            "extra": "mean: 613.388530644438 usec\nrounds: 1566"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed[True-True]",
            "value": 122.27652988017225,
            "unit": "iter/sec",
            "range": "stddev: 0.00006493292265953568",
            "extra": "mean: 8.17818432515196 msec\nrounds: 120"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed[True-False]",
            "value": 122.11904590208435,
            "unit": "iter/sec",
            "range": "stddev: 0.00007980873370260783",
            "extra": "mean: 8.188730861865764 msec\nrounds: 123"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed[False-True]",
            "value": 122.17692096413943,
            "unit": "iter/sec",
            "range": "stddev: 0.0001269731115327033",
            "extra": "mean: 8.1848518698021 msec\nrounds: 123"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed[False-False]",
            "value": 122.84959587005048,
            "unit": "iter/sec",
            "range": "stddev: 0.0000901004593267835",
            "extra": "mean: 8.140034917638587 msec\nrounds: 121"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-True]",
            "value": 50.283640800976826,
            "unit": "iter/sec",
            "range": "stddev: 0.00013419083812433598",
            "extra": "mean: 19.887183665916528 msec\nrounds: 51"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-False]",
            "value": 50.257655083939554,
            "unit": "iter/sec",
            "range": "stddev: 0.00009514702122326463",
            "extra": "mean: 19.89746633283657 msec\nrounds: 51"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-True]",
            "value": 50.43376537755754,
            "unit": "iter/sec",
            "range": "stddev: 0.00013497527390772383",
            "extra": "mean: 19.827986122269362 msec\nrounds: 49"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-False]",
            "value": 50.36706002333122,
            "unit": "iter/sec",
            "range": "stddev: 0.00012856667190197358",
            "extra": "mean: 19.854246000000323 msec\nrounds: 51"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 649.5875572947484,
            "unit": "iter/sec",
            "range": "stddev: 0.00003165965151381988",
            "extra": "mean: 1.5394383540296985 msec\nrounds: 624"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 662.4073113641699,
            "unit": "iter/sec",
            "range": "stddev: 0.000016320129097879864",
            "extra": "mean: 1.509645172757208 msec\nrounds: 602"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vmoens@meta.com",
            "name": "Vincent Moens",
            "username": "vmoens"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fd5e0cf5c500fbb726417f15e7c3966b5cf85e93",
          "message": "[BugFix] Fix vmap for tensorclass (#778)",
          "timestamp": "2024-05-15T11:34:03+01:00",
          "tree_id": "ef3c8d9d9b88a1e9219b0dfce98669f1b8267b50",
          "url": "https://github.com/pytorch/tensordict/commit/fd5e0cf5c500fbb726417f15e7c3966b5cf85e93"
        },
        "date": 1715769659437,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested",
            "value": 74456.5686722682,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011476405053704813",
            "extra": "mean: 13.430648468393038 usec\nrounds: 14514"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested",
            "value": 74240.16738324703,
            "unit": "iter/sec",
            "range": "stddev: 7.70116984716138e-7",
            "extra": "mean: 13.469797216885846 usec\nrounds: 20577"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_nested_inplace",
            "value": 68182.11157985505,
            "unit": "iter/sec",
            "range": "stddev: 8.022994626010824e-7",
            "extra": "mean: 14.666603553760545 usec\nrounds: 38109"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_plain_set_stack_nested_inplace",
            "value": 67694.52643803613,
            "unit": "iter/sec",
            "range": "stddev: 8.225390013215703e-7",
            "extra": "mean: 14.772243084015743 usec\nrounds: 43383"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items",
            "value": 211427.0867463325,
            "unit": "iter/sec",
            "range": "stddev: 3.423913097511486e-7",
            "extra": "mean: 4.729762942814357 usec\nrounds: 88818"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested",
            "value": 2963.837738054684,
            "unit": "iter/sec",
            "range": "stddev: 0.0000050768071091208145",
            "extra": "mean: 337.400387059094 usec\nrounds: 2580"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_locked",
            "value": 2951.2328510953084,
            "unit": "iter/sec",
            "range": "stddev: 0.000004041996722623188",
            "extra": "mean: 338.8414437135532 usec\nrounds: 2823"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_nested_leaf",
            "value": 12073.841820506324,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032611247430719695",
            "extra": "mean: 82.82367906307923 usec\nrounds: 9133"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested",
            "value": 2956.2484459371362,
            "unit": "iter/sec",
            "range": "stddev: 0.000004334300660140052",
            "extra": "mean: 338.26656260042387 usec\nrounds: 2729"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_leaf",
            "value": 11820.612034104826,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019423178519789703",
            "extra": "mean: 84.59798842181777 usec\nrounds: 9691"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_items_stack_nested_locked",
            "value": 2955.4771854507003,
            "unit": "iter/sec",
            "range": "stddev: 0.0000043986759473305826",
            "extra": "mean: 338.35483654647237 usec\nrounds: 2822"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys",
            "value": 210304.51795697302,
            "unit": "iter/sec",
            "range": "stddev: 3.5905006288323105e-7",
            "extra": "mean: 4.755009591399238 usec\nrounds: 119775"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested",
            "value": 14579.48463181034,
            "unit": "iter/sec",
            "range": "stddev: 0.000001916800024686143",
            "extra": "mean: 68.58953010027143 usec\nrounds: 10787"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_locked",
            "value": 13518.462828405301,
            "unit": "iter/sec",
            "range": "stddev: 0.000019638693105982446",
            "extra": "mean: 73.9729074742712 usec\nrounds: 10563"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_nested_leaf",
            "value": 16984.550363155086,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017546729639487177",
            "extra": "mean: 58.87703699058877 usec\nrounds: 13864"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested",
            "value": 14612.390692024279,
            "unit": "iter/sec",
            "range": "stddev: 0.000001946410010066956",
            "extra": "mean: 68.43507137718532 usec\nrounds: 12541"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_leaf",
            "value": 17107.4252906996,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016646434748297222",
            "extra": "mean: 58.45414976288962 usec\nrounds: 14141"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_keys_stack_nested_locked",
            "value": 13634.324164360274,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025921973254078462",
            "extra": "mean: 73.34430280115906 usec\nrounds: 12502"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values",
            "value": 550724.2549548394,
            "unit": "iter/sec",
            "range": "stddev: 1.876803471746388e-7",
            "extra": "mean: 1.8157907355687506 usec\nrounds: 173611"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested",
            "value": 27946.40303135007,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013891407314385278",
            "extra": "mean: 35.78278030550863 usec\nrounds: 19740"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_locked",
            "value": 26617.653798111973,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015223489774088612",
            "extra": "mean: 37.56905126142002 usec\nrounds: 23883"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_nested_leaf",
            "value": 31626.43319355965,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013279394218419632",
            "extra": "mean: 31.61912043257658 usec\nrounds: 24938"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested",
            "value": 27500.724647310817,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016050736896627491",
            "extra": "mean: 36.36267817756526 usec\nrounds: 21075"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_leaf",
            "value": 30776.43869104042,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013001266226440896",
            "extra": "mean: 32.492388415658965 usec\nrounds: 22017"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_values_stack_nested_locked",
            "value": 26263.252761259442,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015070753862069103",
            "extra": "mean: 38.076014768250104 usec\nrounds: 23585"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership",
            "value": 1410399.679878151,
            "unit": "iter/sec",
            "range": "stddev: 3.282451889722425e-8",
            "extra": "mean: 709.0188790218622 nsec\nrounds: 67381"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested",
            "value": 387334.9784958771,
            "unit": "iter/sec",
            "range": "stddev: 3.0886975549865523e-7",
            "extra": "mean: 2.5817446280820318 usec\nrounds: 152672"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf",
            "value": 383145.0021448215,
            "unit": "iter/sec",
            "range": "stddev: 3.114566673343942e-7",
            "extra": "mean: 2.6099779310758677 usec\nrounds: 195275"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested",
            "value": 393318.2781967721,
            "unit": "iter/sec",
            "range": "stddev: 3.514824936918441e-7",
            "extra": "mean: 2.542470196362735 usec\nrounds: 65920"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf",
            "value": 387472.84072962264,
            "unit": "iter/sec",
            "range": "stddev: 3.1492911452003375e-7",
            "extra": "mean: 2.5808260473610765 usec\nrounds: 126407"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_last",
            "value": 323347.2303878597,
            "unit": "iter/sec",
            "range": "stddev: 3.4409323257072236e-7",
            "extra": "mean: 3.092650581235799 usec\nrounds: 148127"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_nested_leaf_last",
            "value": 323510.4576178229,
            "unit": "iter/sec",
            "range": "stddev: 3.561373012480011e-7",
            "extra": "mean: 3.091090184111896 usec\nrounds: 148346"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_last",
            "value": 326194.5741997886,
            "unit": "iter/sec",
            "range": "stddev: 3.6056826872170005e-7",
            "extra": "mean: 3.0656549160977673 usec\nrounds: 60093"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_membership_stacked_nested_leaf_last",
            "value": 322347.50548575673,
            "unit": "iter/sec",
            "range": "stddev: 3.6018287477335986e-7",
            "extra": "mean: 3.10224209271626 usec\nrounds: 102881"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getleaf",
            "value": 118491.01152271495,
            "unit": "iter/sec",
            "range": "stddev: 5.579616472523401e-7",
            "extra": "mean: 8.43945871631198 usec\nrounds: 83410"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_get",
            "value": 125635.64306955926,
            "unit": "iter/sec",
            "range": "stddev: 5.326496588495074e-7",
            "extra": "mean: 7.959524666470178 usec\nrounds: 90745"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getleaf",
            "value": 119254.49065307797,
            "unit": "iter/sec",
            "range": "stddev: 6.580221392258803e-7",
            "extra": "mean: 8.385428460795575 usec\nrounds: 48193"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_get",
            "value": 126222.2823853204,
            "unit": "iter/sec",
            "range": "stddev: 5.164471343687288e-7",
            "extra": "mean: 7.922531435038443 usec\nrounds: 72307"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitemleaf",
            "value": 116609.75303760507,
            "unit": "iter/sec",
            "range": "stddev: 4.999842409547409e-7",
            "extra": "mean: 8.575612021727835 usec\nrounds: 58137"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_nested_getitem",
            "value": 123004.76951422649,
            "unit": "iter/sec",
            "range": "stddev: 4.907954544398323e-7",
            "extra": "mean: 8.129766056627112 usec\nrounds: 83271"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitemleaf",
            "value": 116860.08201774642,
            "unit": "iter/sec",
            "range": "stddev: 5.121008134143269e-7",
            "extra": "mean: 8.557241983178992 usec\nrounds: 56754"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_stacked_getitem",
            "value": 124276.77318555964,
            "unit": "iter/sec",
            "range": "stddev: 9.187978326084124e-7",
            "extra": "mean: 8.046555879809366 usec\nrounds: 70373"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_nested",
            "value": 2418.7795068486107,
            "unit": "iter/sec",
            "range": "stddev: 0.0017598787436602422",
            "extra": "mean: 413.4316489653429 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_lock_stack_nested",
            "value": 3229.640297271384,
            "unit": "iter/sec",
            "range": "stddev: 0.00000598025004917693",
            "extra": "mean: 309.6320047916379 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_nested",
            "value": 2808.6745403633313,
            "unit": "iter/sec",
            "range": "stddev: 0.00007695097011604583",
            "extra": "mean: 356.0398279078072 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unlock_stack_nested",
            "value": 3154.1295307951896,
            "unit": "iter/sec",
            "range": "stddev: 0.000006283624448204908",
            "extra": "mean: 317.04468387761153 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_flatten_speed",
            "value": 9698.256198169427,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036889550828237498",
            "extra": "mean: 103.11132017617278 usec\nrounds: 6795"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unflatten_speed",
            "value": 3419.4771319465667,
            "unit": "iter/sec",
            "range": "stddev: 0.000004716310444278591",
            "extra": "mean: 292.44237098633306 usec\nrounds: 3151"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_common_ops",
            "value": 1655.6442245519354,
            "unit": "iter/sec",
            "range": "stddev: 0.00006505271741058443",
            "extra": "mean: 603.9944966260059 usec\nrounds: 1098"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation",
            "value": 612181.4910092569,
            "unit": "iter/sec",
            "range": "stddev: 2.5137099360720384e-7",
            "extra": "mean: 1.633502506505671 usec\nrounds: 146178"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_empty",
            "value": 102147.81338270528,
            "unit": "iter/sec",
            "range": "stddev: 6.520106466989657e-7",
            "extra": "mean: 9.789734766552632 usec\nrounds: 31289"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_1",
            "value": 86748.41887651963,
            "unit": "iter/sec",
            "range": "stddev: 9.023109867795947e-7",
            "extra": "mean: 11.527587625815185 usec\nrounds: 35136"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_creation_nested_2",
            "value": 73270.06238045594,
            "unit": "iter/sec",
            "range": "stddev: 8.976691088184319e-7",
            "extra": "mean: 13.64813905586001 usec\nrounds: 31877"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_clone",
            "value": 78934.20710106917,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017520889661470965",
            "extra": "mean: 12.668778679433329 usec\nrounds: 17477"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[int]",
            "value": 88757.68313048856,
            "unit": "iter/sec",
            "range": "stddev: 8.003144170997834e-7",
            "extra": "mean: 11.2666302761625 usec\nrounds: 13621"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[slice_int]",
            "value": 47210.69881714072,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014121628281563935",
            "extra": "mean: 21.181639438832697 usec\nrounds: 11783"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[range]",
            "value": 19889.3345571889,
            "unit": "iter/sec",
            "range": "stddev: 0.000011805089298727692",
            "extra": "mean: 50.27820297982544 usec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[tuple]",
            "value": 51498.89591810294,
            "unit": "iter/sec",
            "range": "stddev: 0.000001154639011817746",
            "extra": "mean: 19.417892018311775 usec\nrounds: 16483"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem[list]",
            "value": 27794.884931781926,
            "unit": "iter/sec",
            "range": "stddev: 0.000003722147181452627",
            "extra": "mean: 35.97784277410535 usec\nrounds: 12294"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[int]",
            "value": 31293.41108068688,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018622555679672917",
            "extra": "mean: 31.955608719727024 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[slice_int]",
            "value": 18982.905020510625,
            "unit": "iter/sec",
            "range": "stddev: 0.000002469086507728756",
            "extra": "mean: 52.678976106108166 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[range]",
            "value": 14336.265969045635,
            "unit": "iter/sec",
            "range": "stddev: 0.00000302953339380358",
            "extra": "mean: 69.75317018805072 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_dim[tuple]",
            "value": 21823.832030761925,
            "unit": "iter/sec",
            "range": "stddev: 0.000002434249398447342",
            "extra": "mean: 45.82146703614853 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem",
            "value": 55379.417600522975,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017678041487907574",
            "extra": "mean: 18.057250208253482 usec\nrounds: 21031"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set",
            "value": 57790.548008780264,
            "unit": "iter/sec",
            "range": "stddev: 0.00000158067961198705",
            "extra": "mean: 17.303867750969715 usec\nrounds: 23370"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_shared",
            "value": 9968.737336446804,
            "unit": "iter/sec",
            "range": "stddev: 0.00006415370359289183",
            "extra": "mean: 100.31360705471592 usec\nrounds: 4774"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update",
            "value": 49539.418752221536,
            "unit": "iter/sec",
            "range": "stddev: 0.000001687049319387819",
            "extra": "mean: 20.185945358011615 usec\nrounds: 30940"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update_nested",
            "value": 40112.9534219769,
            "unit": "iter/sec",
            "range": "stddev: 0.000001900592947021114",
            "extra": "mean: 24.929602901094903 usec\nrounds: 22832"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_update__nested",
            "value": 42187.80162306682,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017750060132037157",
            "extra": "mean: 23.7035342332992 usec\nrounds: 13260"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested",
            "value": 53766.17995135515,
            "unit": "iter/sec",
            "range": "stddev: 0.000001635397421917079",
            "extra": "mean: 18.599052432304994 usec\nrounds: 22712"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_set_nested_new",
            "value": 47753.4644741954,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015764596736882842",
            "extra": "mean: 20.94088902262518 usec\nrounds: 19022"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select",
            "value": 29668.028270294337,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021033032002284965",
            "extra": "mean: 33.70631815803103 usec\nrounds: 12920"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_select_nested",
            "value": 18512.503638881244,
            "unit": "iter/sec",
            "range": "stddev: 0.000005505521120882513",
            "extra": "mean: 54.0175450877282 usec\nrounds: 13573"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_exclude_nested",
            "value": 9113.173826795772,
            "unit": "iter/sec",
            "range": "stddev: 0.000003188700495982558",
            "extra": "mean: 109.73125488506172 usec\nrounds: 5785"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[True]",
            "value": 2884.441221780497,
            "unit": "iter/sec",
            "range": "stddev: 0.0000056889198790853755",
            "extra": "mean: 346.68759843292065 usec\nrounds: 2137"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_empty[False]",
            "value": 1148556.9132515013,
            "unit": "iter/sec",
            "range": "stddev: 4.9868731843994354e-8",
            "extra": "mean: 870.6577692950846 nsec\nrounds: 109410"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to",
            "value": 12254.375677268608,
            "unit": "iter/sec",
            "range": "stddev: 0.00001360363389528846",
            "extra": "mean: 81.60350444086362 usec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_nonblocking",
            "value": 16244.400119307338,
            "unit": "iter/sec",
            "range": "stddev: 0.000002960470302628998",
            "extra": "mean: 61.55967549773946 usec\nrounds: 8837"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed",
            "value": 3695.5874638774726,
            "unit": "iter/sec",
            "range": "stddev: 0.00004586506612204939",
            "extra": "mean: 270.59297331601596 usec\nrounds: 965"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack0",
            "value": 3698.512853316131,
            "unit": "iter/sec",
            "range": "stddev: 0.0000049127311783314125",
            "extra": "mean: 270.37894409462115 usec\nrounds: 3378"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_unbind_speed_stack1",
            "value": 1213.40460336342,
            "unit": "iter/sec",
            "range": "stddev: 0.002912456956193798",
            "extra": "mean: 824.1274157260599 usec\nrounds: 1323"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_split",
            "value": 657.694791916641,
            "unit": "iter/sec",
            "range": "stddev: 0.000013584466616864768",
            "extra": "mean: 1.52046209319344 msec\nrounds: 612"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_chunk",
            "value": 610.9831175380849,
            "unit": "iter/sec",
            "range": "stddev: 0.002851130097282992",
            "extra": "mean: 1.6367064347529474 msec\nrounds: 630"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation[device0]",
            "value": 17067.1233112599,
            "unit": "iter/sec",
            "range": "stddev: 0.000005161313792764974",
            "extra": "mean: 58.592182277153746 usec\nrounds: 3171"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_creation_from_tensor",
            "value": 18653.305839869226,
            "unit": "iter/sec",
            "range": "stddev: 0.000005283815142189781",
            "extra": "mean: 53.609800245842685 usec\nrounds: 6889"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_add_one[memmap_tensor0]",
            "value": 131303.3980399639,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017817789280428273",
            "extra": "mean: 7.615949129478255 usec\nrounds: 25291"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_contiguous[memmap_tensor0]",
            "value": 1418432.478068316,
            "unit": "iter/sec",
            "range": "stddev: 1.7879784537541033e-7",
            "extra": "mean: 705.0035976064537 nsec\nrounds: 88810"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_stack[memmap_tensor0]",
            "value": 199858.50497910482,
            "unit": "iter/sec",
            "range": "stddev: 6.24343488710288e-7",
            "extra": "mean: 5.003539879899281 usec\nrounds: 25893"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index",
            "value": 3429.202968803681,
            "unit": "iter/sec",
            "range": "stddev: 0.0000686191680784878",
            "extra": "mean: 291.6129517842048 usec\nrounds: 785"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_astensor",
            "value": 2744.6927997159414,
            "unit": "iter/sec",
            "range": "stddev: 0.00006702504374886522",
            "extra": "mean: 364.33949916125175 usec\nrounds: 2578"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_memmaptd_index_op",
            "value": 1416.0560180460948,
            "unit": "iter/sec",
            "range": "stddev: 0.00006895220551845135",
            "extra": "mean: 706.186751975972 usec\nrounds: 1045"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 8.650723971553187,
            "unit": "iter/sec",
            "range": "stddev: 0.029953821886599934",
            "extra": "mean: 115.59726137238613 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 0.8083394384884953,
            "unit": "iter/sec",
            "range": "stddev: 0.22045890675940075",
            "extra": "mean: 1.2371040585003357 sec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 8.861198676855293,
            "unit": "iter/sec",
            "range": "stddev: 0.029611192845153452",
            "extra": "mean: 112.85154937468178 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 11.680830174898695,
            "unit": "iter/sec",
            "range": "stddev: 0.03563736241086096",
            "extra": "mean: 85.61035346177121 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 0.809037714189976,
            "unit": "iter/sec",
            "range": "stddev: 0.22906204564300864",
            "extra": "mean: 1.2360363212501397 sec\nrounds: 8"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_pytree",
            "value": 42554.85327155275,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027345371105634048",
            "extra": "mean: 23.499082316622253 usec\nrounds: 10514"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_reshape_td",
            "value": 32511.826049669657,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015665349721583998",
            "extra": "mean: 30.758038581784326 usec\nrounds: 8732"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_pytree",
            "value": 43243.547740557224,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012498091042545203",
            "extra": "mean: 23.12483716645017 usec\nrounds: 20674"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_view_td",
            "value": 28752.312577769928,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025225065258812694",
            "extra": "mean: 34.77981109502676 usec\nrounds: 13826"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_pytree",
            "value": 33850.29048492156,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017238036456534128",
            "extra": "mean: 29.541843974587007 usec\nrounds: 14410"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_unbind_td",
            "value": 23716.722405011562,
            "unit": "iter/sec",
            "range": "stddev: 0.000004930317630077934",
            "extra": "mean: 42.16434222752006 usec\nrounds: 14069"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_pytree",
            "value": 31328.102466061548,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017027419617304671",
            "extra": "mean: 31.92022246107382 usec\nrounds: 9619"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_split_td",
            "value": 25332.800879685034,
            "unit": "iter/sec",
            "range": "stddev: 0.000004846650998984084",
            "extra": "mean: 39.47451388219466 usec\nrounds: 11346"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_pytree",
            "value": 27337.675445341796,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028089486375133863",
            "extra": "mean: 36.57955490763554 usec\nrounds: 8909"
          },
          {
            "name": "benchmarks/common/pytree_benchmarks_test.py::test_add_td",
            "value": 17552.64115681375,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032437134542336353",
            "extra": "mean: 56.971483155502824 usec\nrounds: 9894"
          },
          {
            "name": "benchmarks/distributed/distributed_benchmark_test.py::test_distributed",
            "value": 15327.80184144152,
            "unit": "iter/sec",
            "range": "stddev: 0.000008210601926371705",
            "extra": "mean: 65.24092693424029 usec\nrounds: 4447"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule",
            "value": 65348.470214659865,
            "unit": "iter/sec",
            "range": "stddev: 7.154267402183073e-7",
            "extra": "mean: 15.30257704143878 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdmodule_dispatch",
            "value": 33124.06608341606,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015974005214104072",
            "extra": "mean: 30.189530400093645 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq",
            "value": 57445.39470920492,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015240652596679372",
            "extra": "mean: 17.407835824997164 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_tdseq_dispatch",
            "value": 29432.693220091,
            "unit": "iter/sec",
            "range": "stddev: 0.000002662533891226966",
            "extra": "mean: 33.97582384059206 usec\nrounds: 1000"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_functorch",
            "value": 646.6659993192953,
            "unit": "iter/sec",
            "range": "stddev: 0.000020622135916575964",
            "extra": "mean: 1.5463933484250558 msec\nrounds: 522"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_instantiation_td",
            "value": 938.7370999864853,
            "unit": "iter/sec",
            "range": "stddev: 0.00009872600900827362",
            "extra": "mean: 1.0652609767041237 msec\nrounds: 908"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functorch",
            "value": 6472.15765144133,
            "unit": "iter/sec",
            "range": "stddev: 0.000006371338226013543",
            "extra": "mean: 154.50797923275292 usec\nrounds: 2566"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_functional_call",
            "value": 6837.979590649336,
            "unit": "iter/sec",
            "range": "stddev: 0.000005152079416935211",
            "extra": "mean: 146.24202759649359 usec\nrounds: 4561"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td",
            "value": 6843.973205015675,
            "unit": "iter/sec",
            "range": "stddev: 0.000008399589066872331",
            "extra": "mean: 146.11395603757475 usec\nrounds: 2565"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_exec_td_decorator",
            "value": 4609.487408857992,
            "unit": "iter/sec",
            "range": "stddev: 0.000013432921356480483",
            "extra": "mean: 216.9438619310063 usec\nrounds: 2954"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-True]",
            "value": 1643.4972163761652,
            "unit": "iter/sec",
            "range": "stddev: 0.000012166959301716922",
            "extra": "mean: 608.4585906418225 usec\nrounds: 1237"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[True-False]",
            "value": 1639.3901649408222,
            "unit": "iter/sec",
            "range": "stddev: 0.000012028635161274507",
            "extra": "mean: 609.9829201037676 usec\nrounds: 1567"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-True]",
            "value": 1866.4342022458527,
            "unit": "iter/sec",
            "range": "stddev: 0.000011027722802866711",
            "extra": "mean: 535.7810089403177 usec\nrounds: 1831"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed[False-False]",
            "value": 1858.5604053001982,
            "unit": "iter/sec",
            "range": "stddev: 0.00001208751567860226",
            "extra": "mean: 538.0508468534161 usec\nrounds: 1822"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1480.6097322342944,
            "unit": "iter/sec",
            "range": "stddev: 0.000021564600965820814",
            "extra": "mean: 675.3974246076062 usec\nrounds: 1168"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1488.4442110249497,
            "unit": "iter/sec",
            "range": "stddev: 0.000014447091222410345",
            "extra": "mean: 671.8424463563839 usec\nrounds: 1467"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1682.0054334912372,
            "unit": "iter/sec",
            "range": "stddev: 0.000013594416968997648",
            "extra": "mean: 594.5284004965193 usec\nrounds: 1653"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1686.750771966348,
            "unit": "iter/sec",
            "range": "stddev: 0.000013014929740437043",
            "extra": "mean: 592.8558128562402 usec\nrounds: 1412"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed[True-True]",
            "value": 122.82352239408408,
            "unit": "iter/sec",
            "range": "stddev: 0.00005266200790185708",
            "extra": "mean: 8.141762917297397 msec\nrounds: 122"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed[True-False]",
            "value": 123.31151040765072,
            "unit": "iter/sec",
            "range": "stddev: 0.00003946828786410856",
            "extra": "mean: 8.109543032066828 msec\nrounds: 123"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed[False-True]",
            "value": 124.06249841271473,
            "unit": "iter/sec",
            "range": "stddev: 0.00005054697421670087",
            "extra": "mean: 8.060453503631146 msec\nrounds: 123"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed[False-False]",
            "value": 124.04097545225066,
            "unit": "iter/sec",
            "range": "stddev: 0.000034839154407491245",
            "extra": "mean: 8.061852112610547 msec\nrounds: 124"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-True]",
            "value": 50.732772630262566,
            "unit": "iter/sec",
            "range": "stddev: 0.0000559031032581156",
            "extra": "mean: 19.711124548384937 msec\nrounds: 51"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-False]",
            "value": 50.767866317030006,
            "unit": "iter/sec",
            "range": "stddev: 0.00008673495444611137",
            "extra": "mean: 19.69749907855693 msec\nrounds: 51"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-True]",
            "value": 51.01907894589134,
            "unit": "iter/sec",
            "range": "stddev: 0.00005599041385863198",
            "extra": "mean: 19.600510645450058 msec\nrounds: 51"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-False]",
            "value": 51.00215850986835,
            "unit": "iter/sec",
            "range": "stddev: 0.00016342199423615003",
            "extra": "mean: 19.60701329545711 msec\nrounds: 51"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[True]",
            "value": 649.6291736398243,
            "unit": "iter/sec",
            "range": "stddev: 0.00001661155739377629",
            "extra": "mean: 1.5393397350015454 msec\nrounds: 619"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_to_module_speed[False]",
            "value": 662.7572935743747,
            "unit": "iter/sec",
            "range": "stddev: 0.000018835399703086366",
            "extra": "mean: 1.5088479745078502 msec\nrounds: 595"
          }
        ]
      }
    ]
  }
}