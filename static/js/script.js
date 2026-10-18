document.addEventListener("DOMContentLoaded", function () {

    console.log("Breast Cancer XAI JavaScript loaded.");

    /* File input preview */

    const imageInput = document.querySelector(
        'input[type="file"]'
    );

    if (imageInput) {

        imageInput.addEventListener("change", function () {

            const file = this.files[0];

            if (!file) {
                return;
            }

            const allowedTypes = [
                "image/jpeg",
                "image/png",
                "image/jpg"
            ];

            if (!allowedTypes.includes(file.type)) {

                alert(
                    "Please upload a JPG, JPEG or PNG image."
                );

                this.value = "";
                return;
            }

            const maxSize = 10 * 1024 * 1024;

            if (file.size > maxSize) {

                alert(
                    "Image size must be less than 10 MB."
                );

                this.value = "";
                return;
            }

            console.log(
                "Selected image:",
                file.name
            );

        });

    }


    /* Confirm logout */

    const logoutLinks = document.querySelectorAll(
        ".logout-btn"
    );

    logoutLinks.forEach(function (link) {

        link.addEventListener("click", function (event) {

            const confirmLogout = confirm(
                "Are you sure you want to logout?"
            );

            if (!confirmLogout) {
                event.preventDefault();
            }

        });

    });


    /* Auto hide alerts */

    const alerts = document.querySelectorAll(
        ".alert"
    );

    alerts.forEach(function (alert) {

        setTimeout(function () {

            alert.style.opacity = "0";

            alert.style.transition =
                "opacity 0.5s ease";

            setTimeout(function () {
                alert.remove();
            }, 500);

        }, 5000);

    });

});