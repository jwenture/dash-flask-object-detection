//

//window.alert("sometext");
//window.addEventListener('load', function () {


/* //this works
document.addEventListener("DOMContentLoaded", function(event){

  alert("It's loaded!")
})
*/

//this works
function waitForElementToDisplay(selector, time) {
        if(document.querySelector(selector)!=null) {
            //alert("The element is displayed, you can put your code instead of this alert.");
            //document.getElementById("nothing").innerHTML = "<h3>Hello JavaScript</h3>;";
            var nuts=document.getElementById('submit-button');
            var vid=document.getElementById('imageout');
            nuts.onclick=function(){

                vid.contentWindow.location.reload();
                 
            };
            
            
            

            //alert (nuts);
            //alert (nuts[0].getAttribute('data-title') );
            //[].forEach.call(nuts, function(nut) {
              // do whatever
            //  if (nut.getAttribute('data-title')=="Download plot as a png"){
                 //alert (nuts);
                    //nut.click();
            //  } else {
                  //alert (nut.getAttribute('data-title'));
            //  }
              //div.style.color = "red";
            //};

            return;
        }
        else {
            setTimeout(function() {
                waitForElementToDisplay(selector, time);
            }, time);
        }
    }

waitForElementToDisplay('#submit-button',50);

/* //this works
waitForElementToDisplay('.modebar-btn',100);
var checkExist = setInterval(function() {
   if ($('#outputsave').length) {
      //console.log("Exists!");
      window.alert("It's Exists!");
      clearInterval(checkExist);
   }
}, 100); // check every 100ms
*/

//checkExist();
//document.querySelectorAll('.modebar-btn')[0].click();
