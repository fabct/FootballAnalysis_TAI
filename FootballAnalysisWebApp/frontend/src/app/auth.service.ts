import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private isLoggedIn = false;

  constructor() { }
  // Méthode pour définir l'état de connexion
  setLoggedIn(value: boolean) {
    this.isLoggedIn = value;
  }

  // Méthode pour vérifier l'état de connexion
  getisLoggedIn(): boolean {
    return this.isLoggedIn;
  }

}
