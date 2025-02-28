from django import forms
from .models import PDFDocument

class PDFDocumentForm(forms.ModelForm):
    class Meta:
        model = PDFDocument
        fields = ['file']


class SearchForm(forms.Form):
    query = forms.CharField(label='Search Query', max_length=255)