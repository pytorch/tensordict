window.BENCHMARK_DATA = {
  "lastUpdate": 1714051064575,
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
      }
    ]
  }
}