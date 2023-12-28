document.addEventListener('DOMContentLoaded', (event) => {
    const buttons = document.querySelectorAll('.predict-button');
    buttons.forEach(button => {
        button.addEventListener('mouseover', function() {
            this.style.opacity = '0.8';
        });
        button.addEventListener('mouseout', function() {
            this.style.opacity = '1';
        });
    });
});
