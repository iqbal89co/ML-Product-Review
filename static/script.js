$(function () {
  $(".load-icon-train").hide();
  $(".table-training").hide();
  console.log("ready!");
});

$("#train-form").on("submit", function (e) {
  e.preventDefault();
  var file_train = new FormData($("#train-form")[0]);
  $(".load-icon-train").show();
  $.ajax({
    data: file_train,
    contentType: false,
    cache: false,
    processData: false,
    // async: false,
    type: "post",
    url: "/training",
  }).done(function (data) {
    $(".load-icon-train").hide();
    $("#cth").html(data.cth);
    $("#cth_lower").html(data.cth_lower);
    $("#cth_punctual").html(data.cth_punctual);
    $("#cth_normalize").html(data.cth_normalize);
    $("#cth_stemmed").html(data.cth_stemmed);
    $("#cth_tokenized").html(data.cth_tokenized);
    $(".table-training").show();
    console.log("data", data);
  });
});
