function loadImage() {
	var xhr = new XMLHttpRequest();
	xhr.open("GET", "static/shots/shot_2023-02-11 183241.090025.jpg", true);
	xhr.responseType = "blob";
	
	xhr.onload = function(e) {
	  if (this.status == 200) {
		var blob = this.response;
		var img = document.createElement("img");
		img.src = URL.createObjectURL(blob);
		document.body.appendChild(img);
	  }
	};
	
	xhr.send();
  }
