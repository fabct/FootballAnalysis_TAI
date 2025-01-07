// app.component.ts

import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';
@Component({
  selector: 'my-app',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {

  showContactForm: boolean = false;
  showAdmin: boolean = false
  buttonAdmin: boolean = true;
  buttonContact: boolean = true;

  mapHtml: string =''

  constructor(private http: HttpClient) {}


  onLoginSuccess(event: boolean): void {
    this.showContactForm = event;
  }

  toggleContactForm(): void {
    this.showContactForm = !this.showContactForm;
    this.buttonAdmin = false;
    this.buttonContact = false;
    this.showAdmin = false;
  }
  ;
  toggleAdmin(): void {
    this.showAdmin = !this.showAdmin;
    this.buttonAdmin = false;
    this.buttonContact = false;
    this.showContactForm = false;
  }


}
