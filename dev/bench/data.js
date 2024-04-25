window.BENCHMARK_DATA = {
  "lastUpdate": 1714059589958,
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
      }
    ]
  }
}