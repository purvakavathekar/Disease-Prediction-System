/* Sign-in / registration: inline validation messages.
   Uses the browser's own constraint rules (required, minlength, min/max, type=email),
   so it enforces exactly what the HTML attributes already declare. It only changes how
   problems are shown; the form still posts the same fields to the same routes. */
(function () {
  'use strict';
  var form = document.querySelector('form[data-validate]');
  if (!form) return;
  form.setAttribute('novalidate', '');

  var summary = document.getElementById('form-summary');
  var fields = Array.prototype.slice.call(form.querySelectorAll('input, select'))
    .filter(function (el) { return el.type !== 'hidden' && el.type !== 'submit'; });

  function label(el) { return el.getAttribute('data-label') || 'This field'; }

  function messageFor(el) {
    var v = el.validity;
    if (v.valueMissing) return label(el) + ' is required.';
    if (v.typeMismatch) return 'Enter a valid email address, like name@example.com.';
    if (v.tooShort) return label(el) + ' needs at least ' + el.getAttribute('minlength') + ' characters.';
    if (v.rangeUnderflow || v.rangeOverflow) return label(el) + ' must be between ' + el.min + ' and ' + el.max + '.';
    if (v.badInput) return 'Enter ' + label(el).toLowerCase() + ' as a number.';
    return el.validationMessage || (label(el) + ' is not valid.');
  }

  function paint(el) {
    var wrap = el.closest('.auth-field');
    var out = document.getElementById(el.id + '-error');
    if (!wrap || !out) return true;
    var ok = el.checkValidity();
    wrap.classList.toggle('invalid', !ok);
    el.setAttribute('aria-invalid', ok ? 'false' : 'true');
    out.textContent = ok ? '' : messageFor(el);
    return ok;
  }

  fields.forEach(function (el) {
    el.addEventListener('blur', function () { if (el.value !== '' || el.closest('.auth-field').classList.contains('invalid')) paint(el); });
    el.addEventListener('input', function () { if (el.closest('.auth-field').classList.contains('invalid')) paint(el); });
    el.addEventListener('change', function () { if (el.closest('.auth-field').classList.contains('invalid')) paint(el); });
  });

  form.addEventListener('submit', function (e) {
    var bad = fields.filter(function (el) { return !paint(el); });
    if (bad.length) {
      e.preventDefault();
      if (summary) {
        summary.textContent = bad.length === 1
          ? 'One field needs attention before you can continue.'
          : bad.length + ' fields need attention before you can continue.';
        summary.classList.add('show');
      }
      bad[0].focus();
    }
  });
})();
