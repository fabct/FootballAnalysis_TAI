import { Component } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-admin',
  templateUrl: './admin.component.html',
  styleUrls: ['./admin.component.css']
})
export class AdminComponent {
 

  isLogin: boolean = false;
  isOTPSendEvent2: boolean = false;

  swapLoginResiter :boolean = false;
  constructor(private router: Router) {}

  onLoginSuccess(event: boolean): void {
    this.isLogin = event;
    this.router.navigate(['/map']); // Assure-toi que 'map' est le chemin configuré pour ton composant Map

  }
  isOTPSendEvent(event: boolean): void {
    this.isOTPSendEvent2 = event;
  }

  toggleView() {
    this.swapLoginResiter = !this.swapLoginResiter; // Inverse la valeur entre true et false
  }



}
